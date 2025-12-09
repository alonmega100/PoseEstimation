import h5py
import numpy as np
from collections import deque
import threading
import os
import time


# REMOVED: from multiprocessing.shared_memory import SharedMemory (Causes heap corruption in threaded apps)

class HDF5Writer:
    def __init__(self, file_name, sample_name):
        self.file_name = file_name
        self.camera_data = deque()
        self.robot_data = deque()
        self.digit_data = deque()
        self.xela_data = deque()
        self.write_enabled = False
        self.to_file_enabled = False
        self.stop_event = threading.Event()
        self.thread = None

        # FIX: Replaced SharedMemory with a simple atomic boolean flag.
        # Threads share memory automatically; SharedMemory is only for separate processes.
        self.write_xela_enabled = False

        # Open or create HDF5 file
        self.hdf5_file = self._initialize_file()
        self.hdf5_file.attrs['sample_name'] = sample_name

    def start_writing(self):
        self.write_enabled = True
        self.write_xela_enabled = True  # Simple boolean assignment

    def stop_writing(self):
        self.write_enabled = False
        self.write_xela_enabled = False

    def _initialize_file(self):
        """Initialize or open an HDF5 file."""
        if os.path.exists(self.file_name):
            return h5py.File(self.file_name, 'a')  # Append mode
        return h5py.File(self.file_name, 'w')  # Create new file

    def add_run(self, run_name, force, dx, dy, angle):
        """Create a new run directly under the root, or reuse if it exists."""
        print(f"Adding new run: {run_name}")

        if run_name in self.hdf5_file:
            print(f"Run '{run_name}' already exists, reusing existing group.")
            self.current_run = self.hdf5_file[run_name]
        else:
            self.current_run = self.hdf5_file.create_group(run_name)

        self.current_run.attrs.update({
            "force": force,
            "dx": dx,
            "dy": dy,
            "angle": angle
        })

        # Set up subgroups inside the run
        self.camera_group = self._ensure_group(self.current_run, 'camera', datasets={
            'frames': (0, 420, 900, 3),
            'timestamps': (0,)
        }, dtype_map={'frames': 'uint8', 'timestamps': 'f8'})

        self.robot_group = self._ensure_group(self.current_run, 'robot', datasets={
            'q': (0, 7), 'q_dot': (0, 7), 'calculated_force': (0, 6),
            'tau': (0, 7), 'ee_position': (0, 3), 'ee_rotation': (0, 3, 3), 'timestamps': (0,)
        }, dtype_map={'q': 'f8', 'q_dot': 'f8', 'calculated_force': 'f8',
                      'tau': 'f8', 'ee_position': 'f8', 'ee_rotation': 'f8', 'timestamps': 'f8'})

        self.digit_group = self._ensure_group(self.current_run, 'digit', datasets={
            'frames': (0, 420, 900, 3),
            'timestamps': (0,)
        }, dtype_map={'frames': 'uint8', 'timestamps': 'f8'})

        self.xela_group = self._ensure_group(self.current_run, 'xela', datasets={
            'data': (0, 90), 'timestamps': (0,)
        }, dtype_map={
            'data': 'f8', 'timestamps': 'f8'
        })

        print(f"Run {run_name} added successfully.")

    def _ensure_group(self, parent_group, group_name, datasets, dtype_map):
        """Ensure a group and its datasets exist in the HDF5 file."""
        if group_name not in parent_group:
            group = parent_group.create_group(group_name)
        else:
            group = parent_group[group_name]

        for dataset_name, shape in datasets.items():
            if dataset_name not in group:
                group.create_dataset(dataset_name, shape=shape, maxshape=(None,) + shape[1:],
                                     dtype=dtype_map[dataset_name])

        return group

    def set_sample_name(self, sample_name, file_name):
        """Switch to a new HDF5 file for a new sample."""
        print(f"Switching to new sample: {sample_name} -> {file_name}")

        self.stop()
        self.file_name = file_name
        self.hdf5_file = self._initialize_file()
        self.hdf5_file.attrs['sample_name'] = sample_name
        self.start()

    def start(self):
        """Start the writing thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the writing thread and close the HDF5 file."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()

        # Close file
        if self.hdf5_file:
            self.hdf5_file.close()

        # REMOVED: SharedMemory cleanup logic (no longer needed)

    def add_camera_data(self, frame, timestamp):
        """Add camera data to the queue, ensuring 10Hz saving."""
        if self.write_enabled and (not self.camera_data or timestamp - self.camera_data[-1][1] >= 0.1):
            self.camera_data.append((frame, timestamp))

    def add_robot_data(self, q, q_dot, calculated_force, tau, position, timestamp):
        """Add robot data to the queue, extracting position & rotation."""
        if self.write_enabled:
            translation = position[0:3, 3]
            rotation = position[0:3, 0:3]
            self.robot_data.append((q, q_dot, calculated_force, tau, translation, rotation, timestamp))

    def add_xela_data(self, xela_array, timestamp):
        """Add XELA data to the queue."""
        self.xela_data.append((xela_array, timestamp))

    def _write_loop(self):
        """Internal loop for writing data to the HDF5 file."""
        written = False
        while not self.stop_event.is_set():
            if self.to_file_enabled:
                t1 = time.time()
                if self.camera_data:
                    written = True
                    frames, timestamps = zip(*self.camera_data)
                    # Fixed slicing logic as per your file
                    cropped_frames = np.array([
                        np.frombuffer(frame, dtype=np.uint8).reshape(720, 1280, 3)[130:550, 200:1100, :]
                        for frame in frames
                    ])
                    self._append_to_dataset(self.camera_group, 'frames', cropped_frames)
                    self._append_to_dataset(self.camera_group, 'timestamps', np.array(timestamps, dtype=np.float64))
                    self.camera_data.clear()
                    print('Camera data written.')

                if self.robot_data:
                    written = True
                    q, q_dot, calculated_force, tau, translation, rotation, timestamps = zip(*self.robot_data)
                    self._append_to_dataset(self.robot_group, 'q', np.array(q))
                    self._append_to_dataset(self.robot_group, 'q_dot', np.array(q_dot))
                    self._append_to_dataset(self.robot_group, 'calculated_force', np.array(calculated_force))
                    self._append_to_dataset(self.robot_group, 'tau', np.array(tau))
                    self._append_to_dataset(self.robot_group, 'ee_position', np.array(translation))
                    self._append_to_dataset(self.robot_group, 'ee_rotation', np.array(rotation))
                    self._append_to_dataset(self.robot_group, 'timestamps', np.array(timestamps))
                    self.robot_data.clear()

                if self.xela_data:
                    written = True
                    xela_values, timestamps = zip(*self.xela_data)
                    self._append_to_dataset(self.xela_group, 'data', np.array(xela_values))
                    self._append_to_dataset(self.xela_group, 'timestamps', np.array(timestamps))
                    self.xela_data.clear()
                    print('XELA sensor data written.')

                if written:
                    print(f"Data written in {time.time() - t1:.6f} seconds.")
                    written = False
            time.sleep(0.1)

    def clear_data(self):
        """Clear all data queues."""
        self.camera_data.clear()
        self.robot_data.clear()
        self.digit_data.clear()
        self.xela_data.clear

    def _append_to_dataset(self, group, name, data):
        """Append data to a dataset, resizing it if necessary."""
        dataset = group[name]
        data = np.array(data)

        num_existing = dataset.shape[0]
        dataset.resize((num_existing + data.shape[0],) + dataset.shape[1:])
        dataset[num_existing:] = data