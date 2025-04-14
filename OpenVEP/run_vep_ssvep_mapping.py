from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle
import mne
import pyautogui

cyton_in = True
lsl_out = False
width = 2880
height = 1800
aspect_ratio = width/height
refresh_rate = 120
stim_duration = 1.2
n_per_class = 2
stim_type = 'alternating' # 'alternating' or 'independent'
subject = 10
session = 1
calibration_mode = False
save_dir = f'data/cyton8_{stim_type}-vep_4-class_{stim_duration}s/sub-{subject:02d}/ses-{session:02d}/' # Directory to save data to
run = 1 # Run number, it is used as the random seed for the trial sequence generation
save_file_eeg = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_timestamp = save_dir + f'timestamp_{n_per_class}-per-class_run-{run}.npy'
save_file_metadata = save_dir + f'metadata_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'
model_file_path = 'cache/FBTRCA_model.pkl'


import string
import numpy as np
import psychopy.visual
import psychopy.event
from psychopy import core

letters = '↑→↓←'

def create_4_targets(size=2/8*0.7, colors=[-1, -1, -1] * 4, checkered=False, elementTex=None, elementMask=None, phases=None):
    positions = create_4_target_positions()

    texture = elementTex
    keys = visual.ElementArrayStim(window, nElements=4, elementTex=texture, elementMask=elementMask, units='norm',
                                   sizes=[size, size * aspect_ratio], xys=positions, phases=phases, colors=colors) # sizes=[size, size * aspect_ratio]
    return keys

def create_4_target_positions():
    positions = [[0,0.5], [0.5,0], [0, -0.5], [-0.5,0]]
    return positions

def create_4_key_caps():
    positions = create_4_target_positions()
    keycaps = visual.ElementArrayStim(window, nElements=4, xys=positions)
    return keycaps

def create_photosensor_dot(size=2/8*0.7):
    width, height = window.size
    ratio = width/height
    return visual.Rect(win=window, units="norm", width=size, height=size * ratio,
                       fillColor='white', lineWidth = 0, pos = [1 - size/2, -1 - size/8]
    )

def create_trial_sequence(n_per_class, classes=[(7.5, 0), (8.57, 0), (10, 0), (12, 0), (15, 0)], seed=0):
    """
    Create a random sequence of trials with n_per_class of each class
    Inputs:
        n_per_class : number of trials for each class
    Outputs:
        seq : (list of len(10 * n_per_class)) the trial sequence
    """
    seq = classes * n_per_class
    random.seed(seed)
    random.shuffle(seq)  # shuffles in-place
    return seq

keyboard = keyboard.Keyboard()
window = visual.Window(
        size = [width,height],
        checkTiming = True,
        allowGUI = True,
        fullscr = True,
        useRetina = False
    )

visual_stimulus = create_4_targets(checkered=False)
key_caps = create_4_key_caps()
photosensor_dot = create_photosensor_dot()
photosensor_dot.color = np.array([-1, -1, -1])
photosensor_dot.draw()
window.flip()

if cyton_in:
    import glob, sys, time, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue
    sampling_rate = 250
    CYTON_BOARD_ID = 0 # 0 if no daisy 2 if use daisy board, 6 if using daisy+wifi shield
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2' # Reads from analog pins A5(D11), A6(D12) and if no
                        # wifi shield is present, then A7(D13) as well.
    def find_openbci_port():
        print("running")
        """Finds the port to which the Cyton Dongle is connected to."""
        # Find serial port names per OS
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
            exit()
        else:
            return openbci_port

    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    res_query = board.config_board('/0')
    print(res_query)
    res_query = board.config_board('//')
    print(res_query)
    res_query = board.config_board(ANALOGUE_MODE)
    print(res_query)
    board.start_stream(45000)
    stop_event = Event()

    def get_data(queue_in, lsl_out=False):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)

    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None

num_frames = np.round(stim_duration * refresh_rate).astype(int)  # total number of frames per trial
frame_indices = np.arange(num_frames)  # frame indices for the trial
if stim_type == 'alternating': # Alternating VEP (aka SSVEP)
    #choose 4
    stimulus_classes = [(8, 0), (10, 0), (12,0), (15,0)] # flickering frequencies (in hz) and phase offsets (in pi*radians)

    stimulus_frames = np.zeros((num_frames, len(stimulus_classes)))
    for i_class, (flickering_freq, phase_offset) in enumerate(stimulus_classes):
            phase_offset += .00001  # nudge phase slightly from points of sudden jumps for offsets that are pi multiples
            stimulus_frames[:, i_class] = signal.square(2 * np.pi * flickering_freq * (frame_indices / refresh_rate) + phase_offset * np.pi)  # frequency approximation formula
# trial_sequence = create_trial_sequence(n_per_class=n_per_class, classes=stimulus_classes, seed=run)
trial_sequence = np.tile(np.arange(4), n_per_class)
np.random.seed(run)
np.random.shuffle(trial_sequence)
target_positions = create_4_target_positions()
eeg = np.zeros((8, 0))
aux = np.zeros((3, 0))
timestamp = np.zeros((0))
eeg_trials = []
aux_trials = []
trial_ends = []
skip_count = 0 # Number of trials to skip due to frame miss in those trials
accuracy = 0
predictions = []
aim_target_color = 'white'

if calibration_mode:
    for i_trial, target_id in enumerate(trial_sequence):
        print(target_id)
        trial_text = visual.TextStim(window, text=f'Trial {i_trial+1}/{len(trial_sequence)}', pos=(0, -1+0.07), color='white', units='norm', height=0.07)
        trial_text.draw()
        accuracy_text = visual.TextStim(window, text=f'Accuracy: {accuracy*100:.2f}%', pos=(0, 1-0.07), color=aim_target_color, units='norm', height=0.07)
        accuracy_text.draw()
        visual_stimulus.colors = np.array([-1] * 3).T
        visual_stimulus.draw()
        key_caps.draw()
        photosensor_dot.color = np.array([-1, -1, -1])
        photosensor_dot.draw()
        aim_target = visual.Rect(win=window, units="norm", width=2/8*0.7 * 1.3, height=2/8*0.7*aspect_ratio * 1.3, pos=target_positions[target_id], lineColor=aim_target_color, lineWidth=3)
        aim_target.draw()
        window.flip()
        core.wait(0.7)
        finished_displaying = False
        while not finished_displaying:
            for i_frame in range(num_frames):
                next_flip = window.getFutureFlipTime()
                keys = keyboard.getKeys()
                if 'escape' in keys:
                    if cyton_in:
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(save_file_eeg, eeg)
                        np.save(save_file_aux, aux)
                        # np.save(save_file_timestamp, timestamp)
                        np.save(save_file_eeg_trials, eeg_trials)
                        np.save(save_file_aux_trials, aux_trials)
                        stop_event.set()
                        board.stop_stream()
                        board.release_session()
                    core.quit()
                visual_stimulus.colors = np.array([stimulus_frames[i_frame]] * 3).T
                visual_stimulus.draw()
                photosensor_dot.color = np.array([1, 1, 1])
                photosensor_dot.draw()
                trial_text.draw()
                aim_target.draw()
                if core.getTime() > next_flip and i_frame != 0:
                    print('Missed frame')
                    # skip_count += 1
                    # visual_stimulus.colors = np.array([-1] * 3).T
                    # visual_stimulus.draw()
                    # window.flip()
                    # core.wait(0.5)
                    # visual_stimulus.colors = np.array([-1] * 3).T
                    # visual_stimulus.draw()
                    # photosensor_dot.color = np.array([-1, -1, -1])
                    # photosensor_dot.draw()
                    # trial_text.draw()
                    # aim_target.draw()
                    # window.flip()
                    # core.wait(0.5)
                    # break
                window.flip()
            finished_displaying = True
        visual_stimulus.colors = np.array([-1] * 3).T
        visual_stimulus.draw()
        photosensor_dot.color = np.array([-1, -1, -1])
        photosensor_dot.draw()
        trial_text.draw()
        window.flip()
        if cyton_in:
            while len(trial_ends) <= i_trial+skip_count: # Wait for the current trial to be collected
                while not queue_in.empty(): # Collect all data from the queue
                    eeg_in, aux_in, timestamp_in = queue_in.get()
                    print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                    eeg = np.concatenate((eeg, eeg_in), axis=1)
                    aux = np.concatenate((aux, aux_in), axis=1)
                    timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
                photo_trigger = (aux[1] > 20).astype(int)
                trial_starts = np.where(np.diff(photo_trigger) == 1)[0]
                trial_ends = np.where(np.diff(photo_trigger) == -1)[0]
            print('total: ',eeg.shape, aux.shape, timestamp.shape)
            baseline_duration = 0.2
            baseline_duration_samples = int(baseline_duration * sampling_rate)
            trial_start = trial_starts[i_trial+skip_count] - baseline_duration_samples
            trial_duration = int(stim_duration * sampling_rate) + baseline_duration_samples
            filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
            trial_eeg = np.copy(filtered_eeg[:, trial_start:trial_start+trial_duration])
            trial_aux = np.copy(aux[:, trial_start:trial_start+trial_duration])
            print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
            baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True)
            trial_eeg -= baseline_average
            eeg_trials.append(trial_eeg)
            aux_trials.append(trial_aux)
            cropped_eeg = trial_eeg[:, baseline_duration_samples:]
            if model is not None:
                prediction = model.predict(cropped_eeg)[0]
                predictions.append(prediction)
                accuracy = np.mean(np.array(predictions) == trial_sequence[:i_trial+1])
                # print(predictions, trial_sequence[:i_trial+1], accuracy)
                print(prediction, target_id, accuracy)
                if prediction == target_id:
                    aim_target_color = 'white'
                else:
                    aim_target_color = 'red'

            # time_window = -int((stim_duration + 0.3) * sampling_rate)
            # trial_eeg = np.copy(eeg[time_window:])
            # trial_aux = np.copy(aux[time_window:])
            # photo_trigger = (trial_aux[1] > 20).astype(int)
        # core.wait(1)
    if cyton_in:
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_file_eeg, eeg)
        np.save(save_file_aux, aux)
        np.save(save_file_eeg_trials, eeg_trials)
        np.save(save_file_aux_trials, aux_trials)
        board.stop_stream()
        board.release_session()

else:
    prediction = 0
    pred_text_string = ''
    shift = True
    # while True:
    for i_trial in range(1000):
        pred_text = visual.TextStim(window, text=pred_text_string, pos=(0.07, 1-0.07), color='white', units='norm', height=0.1, alignText='left', wrapWidth=1.94)
        pred_text.draw()
        visual_stimulus.colors = np.array([-1] * 3).T
        visual_stimulus.draw()
        #key_caps.draw()
        for i_letter, position in enumerate(target_positions):
            text = visual.TextStim(win=window, text=letters[i_letter].upper(), pos=position)
            text.draw()
        pred_target = visual.Rect(win=window, units="norm", width=2/8*0.7 * 1.3, height=2/8*0.7*aspect_ratio * 1.3, pos=target_positions[prediction], lineColor='white', lineWidth=3)
        pred_target.draw()
        photosensor_dot.color = np.array([-1, -1, -1])
        photosensor_dot.draw()
        window.flip()
        core.wait(0.7)
        for i_frame in range(num_frames):
            next_flip = window.getFutureFlipTime()

            keys = keyboard.getKeys()
            if 'escape' in keys:
                stop_event.set()
                board.stop_stream()
                board.release_session()
                core.quit()

            visual_stimulus.colors = np.array([stimulus_frames[i_frame]] * 3).T
            visual_stimulus.draw()
            photosensor_dot.color = np.array([1, 1, 1])
            photosensor_dot.draw()
            if core.getTime() > next_flip and i_frame != 0:
                print('Missed frame')
            window.flip()
        visual_stimulus.colors = np.array([-1] * 3).T
        visual_stimulus.draw()
        photosensor_dot.color = np.array([-1, -1, -1])
        photosensor_dot.draw()
        window.flip()
        if cyton_in:
            while len(trial_ends) <= i_trial+skip_count: # Wait for the current trial to be collected
                while not queue_in.empty(): # Collect all data from the queue
                    eeg_in, aux_in, timestamp_in = queue_in.get()
                    print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                    eeg = np.concatenate((eeg, eeg_in), axis=1)
                    aux = np.concatenate((aux, aux_in), axis=1)
                    timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
                photo_trigger = (aux[1] > 20).astype(int)
                trial_starts = np.where(np.diff(photo_trigger) == 1)[0]
                trial_ends = np.where(np.diff(photo_trigger) == -1)[0]
            print('total: ',eeg.shape, aux.shape, timestamp.shape)
            baseline_duration = 0.2
            baseline_duration_samples = int(baseline_duration * sampling_rate)
            trial_start = trial_starts[i_trial+skip_count] - baseline_duration_samples
            trial_duration = int(stim_duration * sampling_rate) + baseline_duration_samples
            filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
            trial_eeg = np.copy(filtered_eeg[:, trial_start:trial_start+trial_duration])
            trial_aux = np.copy(aux[:, trial_start:trial_start+trial_duration])
            print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
            baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True)
            trial_eeg -= baseline_average
            eeg_trials.append(trial_eeg)
            aux_trials.append(trial_aux)
            cropped_eeg = trial_eeg[:, baseline_duration_samples:]
            if model is not None:
                prediction = model.predict(cropped_eeg)[0]

        pred_letter = letters[prediction]
        screen_width, screen_height = pyautogui.size() # Get the size of the primary monitor.
        current_x, current_y = pyautogui.position() # Get the XY position of the mouse.

        if len(pred_text_string) > 74:
            pred_text_string = pred_text_string[-74:]
        if pred_letter == '←':
            print("Moving left")
            new_x = max(current_x - 200, 0)
            pyautogui.moveTo(new_x, current_y, duration=1)
        elif pred_letter == '→':
            print("Moving right")
            screen_width, _ = pyautogui.size()
            new_x = min(current_x + 200, screen_width)
            pyautogui.moveTo(new_x, current_y, duration=1)
        elif pred_letter == '↑':
            print("Moving up")
            new_y = max(current_y - 200, 0)
            pyautogui.moveTo(current_x, new_y, duration=1)
        elif pred_letter == '↓':
            print("Moving down")
            screen_height = pyautogui.size().height
            new_y = min(current_y + 200, screen_height)
            pyautogui.moveTo(current_x, new_y, duration=1)
            
        pyautogui.moveTo(screen_width // 2, screen_height // 2, duration=0.5)
    stop_event.set()
    board.stop_stream()
    board.release_session()