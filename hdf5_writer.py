import h5py
import numpy as np
from collections import deque
import threading
import os
import time
from multiprocessing.shared_memory import SharedMemory

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
        self.shm = SharedMemory(create=True, size=1)  # Create 1-byte shared memory
        self.write_xela = np.ndarray((1,), dtype=np.uint8, buffer=self.shm.buf)  # Use NumPy for lock-free access
        self.write_xela[0] = 0  # Default: False (0)

        # Open or create HDF5 file
        self.hdf5_file = self._initialize_file()
        self.hdf5_file.attrs['sample_name'] = sample_name
        # self.camera_group = self.hdf5_file['camera']
        # self.robot_group = self.hdf5_file['robot']
        # self.digit_group = self.hdf5_file['digit']
        # self.xela_group = self.hdf5_file['xela']

        # self.start()
    
    def start_writing(self):
        self.write_enabled = True
        self.write_xela[0] = 1  # Enable XELA writing
        
    def stop_writing(self):
        self.write_enabled = False
        self.write_xela[0] = 0  # Enable XELA writing
        
    
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

    # def _initialize_file(self):
    #     """Initialize or open an HDF5 file with the required structure."""
    #     if os.path.exists(self.file_name):
    #         hdf5_file = h5py.File(self.file_name, 'a')  # Open in append mode
    #     else:
    #         hdf5_file = h5py.File(self.file_name, 'w')

    #     self._ensure_group(hdf5_file, 'camera', datasets={
    #         'frames': (0, 420, 900, 3),
    #         'timestamps': (0,)
    #     }, dtype_map={'frames': 'uint8', 'timestamps': 'f8'})

    #     self._ensure_group(hdf5_file, 'robot', datasets={
    #         'q': (0, 7), 'q_dot': (0, 7), 'calculated_force': (0, 6),
    #         'tau': (0, 7), 'position': (0, 3), 'rotation': (0, 3, 3), 'timestamps': (0,)
    #     }, dtype_map={'q': 'f8', 'q_dot': 'f8', 'calculated_force': 'f8',
    #                   'tau': 'f8', 'position': 'f8', 'rotation': 'f8', 'timestamps': 'f8'})

    #     self._ensure_group(hdf5_file, 'digit', datasets={
    #         'frames': (0, 420, 900, 3),
    #         'timestamps': (0,)
    #     }, dtype_map={'frames': 'uint8', 'timestamps': 'f8'})
        
    #     self._ensure_group(hdf5_file, 'xela', datasets={
    #         'data': (0, 90), 'timestamps': (0,)
    #     }, dtype_map={
    #         'data': 'f8', 'timestamps': 'f8'
    #     })
    #     # ✅ Only the main 'xela' group contains 'timestamps'
    #     # self._ensure_group(hdf5_file, 'xela', datasets={
    #     #     'timestamps': (0,)
    #     # }, dtype_map={'timestamps': 'f8'})

    #     # # ✅ Each 'xela/xela_{i+1}' group only contains 'data'
    #     # xela_shapes = [4, 5, 6, 6, 5, 4]
    #     # for i in range(6):
    #     #     self._ensure_group(hdf5_file, f'xela/xela_{i+1}', datasets={
    #     #         'data': (0, 3, xela_shapes[i])
    #     #     }, dtype_map={'data': 'f8'})

    #     return hdf5_file

    # def _ensure_group(self, hdf5_file, group_name, datasets, dtype_map):
    #     """Ensure a group and its datasets exist in the HDF5 file."""
    #     if group_name not in hdf5_file:
    #         group = hdf5_file.create_group(group_name)
    #     else:
    #         group = hdf5_file[group_name]

    #     for dataset_name, shape in datasets.items():
    #         if dataset_name not in group:
    #             group.create_dataset(dataset_name, shape=shape, maxshape=(None,) + shape[1:], dtype=dtype_map[dataset_name])

    def _ensure_group(self, parent_group, group_name, datasets, dtype_map):
        """Ensure a group and its datasets exist in the HDF5 file."""
        if group_name not in parent_group:
            group = parent_group.create_group(group_name)
        else:
            group = parent_group[group_name]

        for dataset_name, shape in datasets.items():
            if dataset_name not in group:
                group.create_dataset(dataset_name, shape=shape, maxshape=(None,) + shape[1:], dtype=dtype_map[dataset_name])

        return group
    
    def set_sample_name(self, sample_name, file_name):
        """Switch to a new HDF5 file for a new sample."""
        print(f"Switching to new sample: {sample_name} -> {file_name}")

        self.stop()
        self.file_name = file_name
        self.hdf5_file = self._initialize_file()
        self.hdf5_file.attrs['sample_name'] = sample_name
        self.start()
        
    #  def set_sample_name(self, sample_name, file_name):
    #     """Change the sample name and update the HDF5 file."""
    #     print(f"Changing sample name to: {sample_name}")

    #     self.stop()  # Stop writing while updating metadata
    #     self.file_name = file_name
    #     self.sample_name = sample_name

    #     with h5py.File(self.file_name, 'a') as hdf5_file:
    #         hdf5_file.attrs['sample_name'] = self.sample_name  # Update the root attribute

    #     self.start()  # Restart writing
    #     print(f"Sample name updated to: {sample_name}")

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
        # close file first
        self.hdf5_file.close()
        # NEW: clean up shared memory
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass

    def add_camera_data(self, frame, timestamp):
        """Add camera data to the queue, ensuring 10Hz saving."""
        if self.write_enabled and (not self.camera_data or timestamp - self.camera_data[-1][1] >= 0.1):
            # t1 = time.time()
            self.camera_data.append((frame, timestamp))
            # print(f"Camera: data added in {time.time()-t1:.6f} seconds.")
        # elif self.write_enabled:
        #     print(f'cam time diff:{timestamp - self.camera_data[-1][1]}')

    def add_robot_data(self, q, q_dot, calculated_force, tau, position, timestamp):
        """Add robot data to the queue, extracting position & rotation."""
        if self.write_enabled:
            translation = position[0:3, 3]  
            rotation = position[0:3, 0:3]  
            self.robot_data.append((q, q_dot, calculated_force, tau, translation, rotation, timestamp))

    def add_xela_data(self, xela_array, timestamp):
        """Add XELA data to the queue."""
        # if self.write_enabled:
        #     t1 = time.time()    
        self.xela_data.append((xela_array, timestamp))
            # print(f"Xela: data added in {time.time()-t1:.6f} seconds.")

    def _write_loop(self):
        """Internal loop for writing data to the HDF5 file."""
        written = False
        while not self.stop_event.is_set():
            if self.to_file_enabled:
                t1 = time.time()
                if self.camera_data:
                    written = True
                    # print('Writing camera data...')
                    frames, timestamps = zip(*self.camera_data)
                    cropped_frames = np.array([
                        np.frombuffer(frame, dtype=np.uint8).reshape(720, 1280, 3)[130:550, 200:1100, :]
                        for frame in frames
                    ])

                    self._append_to_dataset(self.camera_group, 'frames', cropped_frames)


                    # self._append_to_dataset(self.camera_group, 'frames', reshaped_frames)
                    # self._append_to_dataset(self.camera_group, 'frames', np.array(frames, dtype=np.uint8))
                    self._append_to_dataset(self.camera_group, 'timestamps', np.array(timestamps, dtype=np.float64))
                    self.camera_data.clear()
                    print('Camera data written.')

                if self.robot_data:
                    written = True
                    # print('Writing robot data...')
                    q, q_dot, calculated_force, tau, translation, rotation, timestamps = zip(*self.robot_data)
                    self._append_to_dataset(self.robot_group, 'q', np.array(q))
                    self._append_to_dataset(self.robot_group, 'q_dot', np.array(q_dot))
                    self._append_to_dataset(self.robot_group, 'calculated_force', np.array(calculated_force))
                    self._append_to_dataset(self.robot_group, 'tau', np.array(tau))
                    self._append_to_dataset(self.robot_group, 'ee_position', np.array(translation))
                    self._append_to_dataset(self.robot_group, 'ee_rotation', np.array(rotation))
                    self._append_to_dataset(self.robot_group, 'timestamps', np.array(timestamps))
                    self.robot_data.clear()
                    # print('Robot data written.')
                if self.xela_data:
                    written = True
                    # print('Writing XELA sensor data...')
                    xela_values, timestamps = zip(*self.xela_data)
                    # xela_shapes = [4, 5, 6, 6, 5, 4]

                    # for i in range(6):
                    #     valid_data = [x[i] for x in xela_values]
                    #     if valid_data:
                    #         subset_data = np.array(valid_data, dtype=np.float64).reshape(-1, 3, xela_shapes[i])
                    #         self._append_to_dataset(self.hdf5_file[f'xela/xela_{i+1}'], 'data', subset_data)
                    
                    self._append_to_dataset(self.xela_group, 'data', np.array(xela_values))
                    self._append_to_dataset(self.xela_group, 'timestamps', np.array(timestamps))
                    self.xela_data.clear()
                    print('XELA sensor data written.')
                if written:
                    print(f"Data written in {time.time()-t1:.6f} seconds.")
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
