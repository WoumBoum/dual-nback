#!/usr/bin/env python3
import pygame
import sys
import random
import os
from collections import deque
import math
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Constants
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
GRID_SIZE = 3
DEFAULT_CELL_SIZE = 100
GRID_PADDING = 50
FPS = 60
RESIZABLE = True

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (30, 144, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Game settings
DEFAULT_N = 2
MIN_N = 1
MAX_N = 9
DEFAULT_TRIAL_DURATION = 3  # seconds
MIN_TRIAL_DURATION = 1.5
MAX_TRIAL_DURATION = 5
DEFAULT_TRIALS_PER_BLOCK = 20
MIN_TRIALS_PER_BLOCK = 5
MAX_TRIALS_PER_BLOCK = 50
DEFAULT_BLOCKS_PER_SESSION = 20
MIN_BLOCKS_PER_SESSION = 1
MAX_BLOCKS_PER_SESSION = 30
LETTERS = ['C', 'H', 'K', 'L', 'Q', 'R', 'S', 'T']

# Implementation of inverse error function to replace scipy.special.erfcinv
def norm_ppf(p):
    """
    Inverse of normal CDF (percent point function)
    A simple approximation of the inverse of the normal CDF
    """
    # Constants for the rational approximation
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924
    
    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857
    
    # Handle edge cases
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    
    # For central part of the distribution
    if 0.02425 <= p <= 0.97575:
        # Central range
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               ((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
    
    # For tails
    if p < 0.02425:
        # Left tail
        q = math.sqrt(-2 * math.log(p))
    else:
        # Right tail
        q = math.sqrt(-2 * math.log(1 - p))
    
    return (((((a1 * q + a2) * q + a3) * q + a4) * q + a5) * q + a6) / \
           ((((b1 * q + b2) * q + b3) * q + b4) * q + b5)

def erfcinv(x):
    """
    Inverse complementary error function
    """
    # Relate erfcinv to the inverse normal CDF
    return -norm_ppf((x) / 2) / math.sqrt(2)

# Dummy sound class to use instead of pygame.mixer.Sound
class DummySound:
    def __init__(self):
        self.volume = 0.5
    
    def play(self):
        # Do nothing, just a placeholder
        pass
    
    def set_volume(self, volume):
        self.volume = volume

class DualNBack:
    def __init__(self):
        # Create resizable window
        flags = pygame.RESIZABLE if RESIZABLE else 0
        self.window_width = DEFAULT_WINDOW_WIDTH
        self.window_height = DEFAULT_WINDOW_HEIGHT
        self.cell_size = DEFAULT_CELL_SIZE
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), flags)
        pygame.display.set_caption("Dual N-Back Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        
        # Define keyboard shortcuts
        self.position_match_key = pygame.K_a  # A key for position match
        self.sound_match_key = pygame.K_l     # L key for sound match
        
        # Make sure pygame mixer is properly initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # Game state
        self.n_level = DEFAULT_N
        self.trial_duration = DEFAULT_TRIAL_DURATION
        self.trial_timer = 0
        self.current_trial = 0
        self.current_block = 0
        self.running = True
        self.paused = False
        self.game_state = "menu"  # menu, playing, block_summary, session_summary, settings, history, paused
        
        # Scrolling state for settings page
        self.settings_scroll_y = 0
        self.settings_content_height = 0
        
        # History view state
        self.history_page = 0  # Which page of history graphs to show
        
        # Settings
        self.show_realtime_score = False
        self.visual_only_mode = True  # Default to visual-only mode (no sound)
        self.trials_per_block = DEFAULT_TRIALS_PER_BLOCK
        self.blocks_per_session = DEFAULT_BLOCKS_PER_SESSION
        
        # Feedback variables
        self.missed_position_match = False   # Flag to show feedback for missed position match
        self.missed_sound_match = False      # Flag to show feedback for missed sound/letter match
        self.feedback_timer = 0              # Timer to control how long feedback is shown
        
        # Performance data for graphs
        self.position_accuracies = []        # List to store position accuracy per block
        self.letter_accuracies = []          # List to store letter accuracy per block
        self.position_dprimes = []           # List to store position d-prime per block  
        self.letter_dprimes = []             # List to store letter d-prime per block
        self.n_levels = []                   # List to store n-level per block
        
        # Graph surfaces
        self.graph_surfaces = {}             # Dictionary to store cached graph surfaces
        
        # Stimulus buffers
        self.position_buffer = deque(maxlen=MAX_N+1)
        self.sound_buffer = deque(maxlen=MAX_N+1)
        
        # Trial data
        self.current_position = None
        self.current_sound = None
        self.current_displayed_letter = None
        self.position_match = False
        self.sound_match = False
        
        # User input
        self.position_response = None
        self.sound_response = None
        
        # Performance metrics
        self.position_correct = 0
        self.position_false_positive = 0
        self.position_false_negative = 0
        self.sound_correct = 0
        self.sound_false_positive = 0
        self.sound_false_negative = 0
        
        # Block results storage
        self.block_results = []
        self.session_results = {}
        
        # Load sounds
        self.sounds = self.load_sounds()
        
        # Load user data if available
        self.user_data = self.load_user_data()
        
    def load_sounds(self):
        sounds = {}
        
        # Create a simple beep sound that we'll use for all letters
        # with different pitch settings for each letter
        for letter in LETTERS:
            # We'll create a dummy sound object to keep the interface consistent
            # but we won't try to play sounds since that's causing issues
            # Instead, we'll just display the letter visually
            sound = DummySound()  # Custom class defined below
            
            # Store the letter with the sound (we'll just use visual feedback)
            sounds[letter] = sound
            
        return sounds
    
    def load_user_data(self):
        try:
            with open('nback_user_data.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'highest_n_level': DEFAULT_N,
                'session_history': [],
                'total_trials': 0,
                'average_accuracy': {'position': 0, 'sound': 0}
            }
    
    def save_user_data(self):
        with open('nback_user_data.json', 'w') as f:
            json.dump(self.user_data, f)
    
    def start_session(self):
        self.n_level = DEFAULT_N
        self.current_trial = 0
        self.current_block = 0
        self.position_buffer.clear()
        self.sound_buffer.clear()
        self.block_results = []
        self.start_block()
        self.game_state = "playing"
    
    def start_block(self):
        self.current_trial = 0
        self.position_correct = 0
        self.position_false_positive = 0
        self.position_false_negative = 0
        self.sound_correct = 0
        self.sound_false_positive = 0
        self.sound_false_negative = 0
        self.position_buffer.clear()
        self.sound_buffer.clear()
        
        # Pre-fill buffers with initial values (no matches in first N trials)
        used_positions = set()
        used_sounds = set()
        
        for _ in range(self.n_level):
            while True:
                pos = random.randint(0, GRID_SIZE * GRID_SIZE - 1)
                if pos not in used_positions:
                    used_positions.add(pos)
                    break
            
            while True:
                sound_idx = random.randint(0, len(LETTERS) - 1)
                if sound_idx not in used_sounds:
                    used_sounds.add(sound_idx)
                    break
            
            self.position_buffer.append(pos)
            self.sound_buffer.append(sound_idx)
        
        self.new_trial()
    
    def new_trial(self):
        # Reset missed match flags when starting a new trial
        # (unless we're still showing feedback)
        if self.feedback_timer <= 0:
            self.missed_position_match = False
            self.missed_sound_match = False
        
        # Decide if this trial will have matches
        if self.current_trial >= self.n_level:
            position_match = random.random() < 0.3  # 30% chance of position match
            sound_match = random.random() < 0.3  # 30% chance of sound match
        else:
            position_match = False
            sound_match = False
        
        if position_match:
            # Use the position from n trials back
            new_position = self.position_buffer[-self.n_level]
        else:
            # Generate a new random position
            while True:
                new_position = random.randint(0, GRID_SIZE * GRID_SIZE - 1)
                # Avoid accidental matches
                if len(self.position_buffer) < self.n_level or new_position != self.position_buffer[-self.n_level]:
                    break
        
        if sound_match:
            # Use the sound from n trials back
            new_sound = self.sound_buffer[-self.n_level]
        else:
            # Generate a new random sound
            while True:
                new_sound = random.randint(0, len(LETTERS) - 1)
                # Avoid accidental matches
                if len(self.sound_buffer) < self.n_level or new_sound != self.sound_buffer[-self.n_level]:
                    break
        
        self.current_position = new_position
        self.current_sound = new_sound
        self.position_match = position_match
        self.sound_match = sound_match
        
        # Reset user responses
        self.position_response = None
        self.sound_response = None
        
        # Visually display the sound letter (we'll skip playing sound since it's causing issues)
        self.current_displayed_letter = LETTERS[new_sound]
        # Use a dummy sound for compatibility with the rest of the code
        try:
            self.sounds[LETTERS[new_sound]].play()
        except:
            pass  # Ignore sound playing errors
        
        # Add to buffers
        self.position_buffer.append(new_position)
        self.sound_buffer.append(new_sound)
        
        # Reset trial timer
        self.trial_timer = 0
    
    def process_user_input(self, position_key=False, sound_key=False):
        # Process position response
        if position_key and self.position_response is None:
            self.position_response = True
            if self.position_match:  # True positive
                self.position_correct += 1
            else:  # False positive
                self.position_false_positive += 1
        
        # Process sound response
        if sound_key and self.sound_response is None:
            self.sound_response = True
            if self.sound_match:  # True positive
                self.sound_correct += 1
            else:  # False positive
                self.sound_false_positive += 1
    
    def end_trial(self):
        # Check for missed matches (false negatives) and set feedback flags
        if self.position_match and self.position_response is None:
            self.position_false_negative += 1
            self.missed_position_match = True  # Set flag for visual feedback
        
        if self.sound_match and self.sound_response is None:
            self.sound_false_negative += 1
            self.missed_sound_match = True  # Set flag for visual feedback
        
        # Reset feedback timer if either type of match was missed
        if self.missed_position_match or self.missed_sound_match:
            self.feedback_timer = 1.0  # Show feedback for 1 second
        
        self.current_trial += 1
        
        if self.current_trial >= self.trials_per_block:
            self.end_block()
        else:
            self.new_trial()
    
    def end_block(self):
        # Calculate accuracy metrics
        position_hits = self.position_correct
        position_false_alarms = self.position_false_positive
        position_misses = self.position_false_negative
        
        sound_hits = self.sound_correct
        sound_false_alarms = self.sound_false_positive
        sound_misses = self.sound_false_negative
        
        # Calculate d-prime (sensitivity index)
        # Add a small constant to avoid division by zero or log of zero
        epsilon = 0.01
        
        # Total number of match trials (signal present)
        position_signal_trials = position_hits + position_misses
        sound_signal_trials = sound_hits + sound_misses
        
        # Total number of non-match trials (signal absent)
        position_noise_trials = self.trials_per_block - position_signal_trials
        sound_noise_trials = self.trials_per_block - sound_signal_trials
        
        # Hit rate and false alarm rate
        position_hit_rate = (position_hits + epsilon) / (position_signal_trials + epsilon)
        position_fa_rate = (position_false_alarms + epsilon) / (position_noise_trials + epsilon)
        
        sound_hit_rate = (sound_hits + epsilon) / (sound_signal_trials + epsilon)
        sound_fa_rate = (sound_false_alarms + epsilon) / (sound_noise_trials + epsilon)
        
        # d-prime calculation
        position_dprime = self.calculate_dprime(position_hit_rate, position_fa_rate)
        sound_dprime = self.calculate_dprime(sound_hit_rate, sound_fa_rate)
        
        # Store block results
        position_accuracy = position_hits / (position_signal_trials + epsilon)
        sound_accuracy = sound_hits / (sound_signal_trials + epsilon)
        
        block_result = {
            'n_level': self.n_level,
            'position_accuracy': position_accuracy,
            'sound_accuracy': sound_accuracy,
            'position_dprime': position_dprime,
            'sound_dprime': sound_dprime
        }
        
        self.block_results.append(block_result)
        
        # Store performance data for graphs
        self.position_accuracies.append(position_accuracy)
        self.letter_accuracies.append(sound_accuracy)
        self.position_dprimes.append(position_dprime)
        self.letter_dprimes.append(sound_dprime)
        self.n_levels.append(self.n_level)
        
        # Update N level based on performance (using both accuracy and d-prime)
        # Using more demanding thresholds of 1.0 and 2.0 for d-prime
        good_dprime = position_dprime >= 2.0 and sound_dprime >= 2.0
        
        if position_accuracy >= 0.8 and sound_accuracy >= 0.8 and good_dprime and self.n_level < MAX_N:
            self.n_level += 1
        elif (position_accuracy <= 0.5 or sound_accuracy <= 0.5 or (position_dprime < 1.0 and sound_dprime < 1.0)) and self.n_level > MIN_N:
            self.n_level -= 1
        
        # Update highest N level achieved
        if self.n_level > self.user_data['highest_n_level']:
            self.user_data['highest_n_level'] = self.n_level
        
        self.current_block += 1
        self.game_state = "block_summary"
        
        if self.current_block >= self.blocks_per_session:
            self.end_session()
    
    def calculate_dprime(self, hit_rate, false_alarm_rate):
        # Create a diagnostic log
        diagnostic = {
            "original_hit_rate": hit_rate,
            "original_fa_rate": false_alarm_rate
        }
        
        # Detect the "didn't respond" case (both rates are 0)
        if hit_rate == 0.0 and false_alarm_rate == 0.0:
            # Return a neutral d-prime of 0 - didn't discriminate at all
            diagnostic["adjusted"] = "No responses case"
            diagnostic["final_dprime"] = 0.0
            self.log_dprime_calculation(diagnostic)
            return 0.0
            
        # Handle perfect performance specifically
        if hit_rate == 1.0 and false_alarm_rate == 0.0:
            diagnostic["adjusted"] = "Perfect performance"
            diagnostic["final_dprime"] = 4.65  # Standard maximum for perfect performance
            self.log_dprime_calculation(diagnostic)
            return 4.65
        
        # For the majority of cases, adjust extreme values
        # Apply the standard corrections for 0 and 1 rates
        # 0 -> 0.5/N and 1 -> (N-0.5)/N where N is the number of trials
        
        # For this implementation, we'll use a simple adjustment method
        # that preserves relative performance levels
        adjusted_hit_rate = hit_rate
        adjusted_fa_rate = false_alarm_rate
        
        if hit_rate == 1.0:
            adjusted_hit_rate = 0.95  # High but not perfect
            diagnostic["hit_rate_adjusted"] = True
            
        elif hit_rate == 0.0:
            adjusted_hit_rate = 0.05  # Low but not zero
            diagnostic["hit_rate_adjusted"] = True
        
        if false_alarm_rate == 0.0:
            adjusted_fa_rate = 0.05  # Low but not zero
            diagnostic["fa_rate_adjusted"] = True
            
        elif false_alarm_rate == 1.0:
            adjusted_fa_rate = 0.95  # High but not perfect
            diagnostic["fa_rate_adjusted"] = True
            
        diagnostic["final_hit_rate"] = adjusted_hit_rate
        diagnostic["final_fa_rate"] = adjusted_fa_rate
        
        # Calculate z-scores for hit rate and false alarm rate
        # using the qnorm function (inverse normal CDF)
        # In standard signal detection theory:
        # d' = Z(hit rate) - Z(false alarm rate)
        try:
            # Use proper z-score conversion (the inverse of the normal CDF)
            from scipy import stats
            z_hit = stats.norm.ppf(adjusted_hit_rate)
            z_fa = stats.norm.ppf(adjusted_fa_rate)
        except (ImportError, NameError):
            # Fall back to our approximation if scipy isn't available
            from math import log, sqrt
            
            # Improved approximation for normal quantile function (inverse CDF)
            def improved_qnorm(p):
                # Abramowitz and Stegun approximation for standard normal quantile
                if p < 0.5:
                    t = sqrt(-2.0 * log(p))
                    return -((0.010328 * t + 0.802853) * t + 2.515517) / ((0.001308 * t + 0.189269) * t + 1.0)
                else:
                    t = sqrt(-2.0 * log(1.0 - p))
                    return ((0.010328 * t + 0.802853) * t + 2.515517) / ((0.001308 * t + 0.189269) * t + 1.0)
            
            z_hit = improved_qnorm(adjusted_hit_rate)
            z_fa = improved_qnorm(adjusted_fa_rate)
            
        diagnostic["z_hit"] = z_hit
        diagnostic["z_fa"] = z_fa
        
        dprime = z_hit - z_fa
        diagnostic["final_dprime"] = dprime
        
        # Log the diagnostic data
        self.log_dprime_calculation(diagnostic)
        
        return dprime
        
    def log_dprime_calculation(self, diagnostic):
        """Log d-prime calculation details for debugging"""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create diagnostic log filename
        log_filename = "logs/dprime_diagnostic.json"
        
        # Load existing log if available
        try:
            with open(log_filename, 'r') as log_file:
                log_data = json.load(log_file)
                if not isinstance(log_data, list):
                    log_data = []
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
        
        # Add timestamp to diagnostic data
        diagnostic["timestamp"] = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Add current N-level
        if hasattr(self, 'n_level'):
            diagnostic["n_level"] = self.n_level
            
        # Add the new diagnostic entry
        log_data.append(diagnostic)
        
        # Limit log size to most recent 100 entries
        if len(log_data) > 100:
            log_data = log_data[-100:]
        
        # Write updated log
        with open(log_filename, 'w') as log_file:
            json.dump(log_data, log_file, indent=2)
    
    def generate_performance_graphs(self):
        """Generate performance graphs for the session"""
        # Clear previous graph surfaces
        self.graph_surfaces = {}
        
        if len(self.block_results) < 2:
            # Not enough data to make meaningful graphs
            return
        
        # Get block numbers for x-axis
        block_numbers = list(range(1, len(self.block_results) + 1))
        
        # Ensure all data arrays have the same length
        # Truncate arrays if needed to match the length of block_results
        position_accuracies = self.position_accuracies[:len(block_numbers)]
        letter_accuracies = self.letter_accuracies[:len(block_numbers)]
        position_dprimes = self.position_dprimes[:len(block_numbers)]
        letter_dprimes = self.letter_dprimes[:len(block_numbers)]
        n_levels = self.n_levels[:len(block_numbers)]
        
        # Create accuracy graph
        fig_accuracy, ax_accuracy = plt.subplots(figsize=(5, 3), dpi=80)
        ax_accuracy.plot(block_numbers, position_accuracies, 'b-o', label='Position Accuracy')
        ax_accuracy.plot(block_numbers, letter_accuracies, 'g-o', label='Letter Accuracy')
        ax_accuracy.set_title('Accuracy by Block')
        ax_accuracy.set_xlabel('Block')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.grid(True, linestyle='--', alpha=0.7)
        ax_accuracy.legend()
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_accuracy)
        canvas.draw()
        
        # Get the renderer and convert it to a format pygame can use
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        
        # Create surface from the matplotlib figure
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['accuracy'] = surf
        plt.close(fig_accuracy)  # Close the figure to free memory
        
        # Create d-prime graph
        fig_dprime, ax_dprime = plt.subplots(figsize=(5, 3), dpi=80)
        ax_dprime.plot(block_numbers, position_dprimes, 'b-o', label='Position d\'')
        ax_dprime.plot(block_numbers, letter_dprimes, 'g-o', label='Letter d\'')
        ax_dprime.set_title('d-prime by Block')
        ax_dprime.set_xlabel('Block')
        ax_dprime.set_ylabel('d-prime')
        ax_dprime.grid(True, linestyle='--', alpha=0.7)
        ax_dprime.legend()
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_dprime)
        canvas.draw()
        
        # Get the renderer and convert it to a format pygame can use
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        
        # Create surface from the matplotlib figure
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['dprime'] = surf
        plt.close(fig_dprime)  # Close the figure to free memory
        
        # Create n-level progression graph
        fig_nlevel, ax_nlevel = plt.subplots(figsize=(5, 3), dpi=80)
        ax_nlevel.plot(block_numbers, n_levels, 'r-o', label='N-Level')
        ax_nlevel.set_title('N-Level Progression')
        ax_nlevel.set_xlabel('Block')
        ax_nlevel.set_ylabel('N-Level')
        ax_nlevel.set_ylim(0, MAX_N + 1)
        ax_nlevel.grid(True, linestyle='--', alpha=0.7)
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_nlevel)
        canvas.draw()
        
        # Get the renderer and convert it to a format pygame can use
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        
        # Create surface from the matplotlib figure
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['n_level'] = surf
        plt.close(fig_nlevel)  # Close the figure to free memory
    
    def end_session(self):
        # Calculate session metrics
        avg_position_accuracy = sum(block['position_accuracy'] for block in self.block_results) / len(self.block_results)
        avg_sound_accuracy = sum(block['sound_accuracy'] for block in self.block_results) / len(self.block_results)
        avg_position_dprime = sum(block['position_dprime'] for block in self.block_results) / len(self.block_results)
        avg_sound_dprime = sum(block['sound_dprime'] for block in self.block_results) / len(self.block_results)
        
        # Get current timestamp for unique identification
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Store session results
        self.session_results = {
            'date': time.strftime("%Y-%m-%d"),
            'timestamp': timestamp,
            'blocks': len(self.block_results),
            'trials_per_block': self.trials_per_block,
            'highest_n_level': max(block['n_level'] for block in self.block_results),
            'final_n_level': self.n_level,
            'avg_position_accuracy': avg_position_accuracy,
            'avg_sound_accuracy': avg_sound_accuracy,
            'position_dprime': avg_position_dprime,
            'sound_dprime': avg_sound_dprime,
            'block_data': self.block_results  # Store detailed block-by-block data
        }
        
        # Update user data
        self.user_data['session_history'].append(self.session_results)
        self.user_data['total_trials'] += self.trials_per_block * self.blocks_per_session
        
        # Update average accuracy
        total_sessions = len(self.user_data['session_history'])
        self.user_data['average_accuracy']['position'] = (
            (self.user_data['average_accuracy']['position'] * (total_sessions - 1) + avg_position_accuracy) / total_sessions
        )
        self.user_data['average_accuracy']['sound'] = (
            (self.user_data['average_accuracy']['sound'] * (total_sessions - 1) + avg_sound_accuracy) / total_sessions
        )
        
        # Generate performance graphs for this session
        self.generate_performance_graphs()
        
        # Save user data
        self.save_user_data()
        
        # Save session log to file in logs directory
        self.save_session_log()
        
        self.game_state = "session_summary"
        
    def save_session_log(self):
        """Save detailed session data to a log file in the logs directory"""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create a filename with timestamp
        filename = f"logs/session_{self.session_results['timestamp']}.json"
        
        # Write session data to file
        with open(filename, 'w') as log_file:
            json.dump(self.session_results, log_file, indent=2)
            
        # Also update the master log that contains a summary of all sessions
        self.update_master_log()
        
    def update_master_log(self):
        """Update the master log file with a summary of all sessions"""
        master_log_file = "logs/master_log.json"
        
        # Create a simplified summary of the current session
        session_summary = {
            'timestamp': self.session_results['timestamp'],
            'date': self.session_results['date'],
            'highest_n_level': self.session_results['highest_n_level'],
            'final_n_level': self.session_results['final_n_level'],
            'avg_position_accuracy': self.session_results['avg_position_accuracy'],
            'avg_sound_accuracy': self.session_results['avg_sound_accuracy'],
            'position_dprime': self.session_results['position_dprime'],
            'sound_dprime': self.session_results['sound_dprime'],
            'blocks': self.session_results['blocks'],
            'trials_per_block': self.session_results['trials_per_block']
        }
        
        # Load existing master log if it exists
        master_log = []
        if os.path.exists(master_log_file):
            try:
                with open(master_log_file, 'r') as f:
                    master_log = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted, start with an empty log
                master_log = []
        
        # Add the new session summary
        master_log.append(session_summary)
        
        # Write the updated master log
        with open(master_log_file, 'w') as f:
            json.dump(master_log, f, indent=2)
            
    def load_history_data(self):
        """Load historical session data from master log"""
        master_log_file = "logs/master_log.json"
        
        if not os.path.exists(master_log_file):
            return []
            
        try:
            with open(master_log_file, 'r') as f:
                history_data = json.load(f)
                
            # Sort history by date
            history_data.sort(key=lambda x: x['timestamp'])
            return history_data
        except (json.JSONDecodeError, KeyError):
            return []
    
    def generate_history_graphs(self):
        """Generate graphs showing progress over time from history data"""
        # Clear previous graph surfaces
        self.graph_surfaces = {}
        
        # Load history data
        history_data = self.load_history_data()
        
        if not history_data or len(history_data) < 2:
            # Not enough data for meaningful graphs
            return False
        
        # Extract data for graphing
        dates = [session['date'] for session in history_data]
        position_accuracies = [session['avg_position_accuracy'] for session in history_data]
        letter_accuracies = [session['avg_sound_accuracy'] for session in history_data]
        position_dprimes = [session['position_dprime'] for session in history_data]
        letter_dprimes = [session['sound_dprime'] for session in history_data]
        n_levels = [session['final_n_level'] for session in history_data]
        
        # Get unique dates for x-axis
        unique_dates = []
        date_indices = {}
        for i, date in enumerate(dates):
            if date not in date_indices:
                date_indices[date] = len(unique_dates)
                unique_dates.append(date)
        
        # Group data by date for proper display
        date_grouped_indices = {date: [] for date in unique_dates}
        for i, date in enumerate(dates):
            date_grouped_indices[date].append(i)
        
        # Create x-axis points based on unique dates
        x_points = list(range(len(unique_dates)))
        
        # For moving average calculation
        window_size = min(3, len(unique_dates))  # Default window size for moving average
        
        # Function to calculate moving average
        def moving_average(data, window_size):
            if len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Create accuracy history graph
        fig_accuracy, ax_accuracy = plt.subplots(figsize=(6, 4), dpi=80)
        
        # Aggregate data by date (average for each day)
        date_pos_accuracies = []
        date_letter_accuracies = []
        
        for date in unique_dates:
            indices = date_grouped_indices[date]
            date_pos_accuracies.append(np.mean([position_accuracies[i] for i in indices]))
            date_letter_accuracies.append(np.mean([letter_accuracies[i] for i in indices]))
        
        # Plot daily data points
        ax_accuracy.plot(x_points, date_pos_accuracies, 'b-o', label='Position Accuracy')
        ax_accuracy.plot(x_points, date_letter_accuracies, 'g-o', label='Letter Accuracy')
        
        # Add moving averages if we have enough data points
        if len(date_pos_accuracies) >= window_size:
            ma_x = x_points[window_size-1:]
            pos_ma = moving_average(date_pos_accuracies, window_size)
            letter_ma = moving_average(date_letter_accuracies, window_size)
            
            ax_accuracy.plot(ma_x, pos_ma, 'b--', alpha=0.7, linewidth=2, label='Position MA')
            ax_accuracy.plot(ma_x, letter_ma, 'g--', alpha=0.7, linewidth=2, label='Letter MA')
        
        ax_accuracy.set_title('Accuracy Progress Over Time')
        ax_accuracy.set_xlabel('Date')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.grid(True, linestyle='--', alpha=0.7)
        ax_accuracy.legend()
        
        # Format x-axis with dates
        ax_accuracy.set_xticks(x_points)
        ax_accuracy.set_xticklabels(unique_dates, rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_accuracy)
        canvas.draw()
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['history_accuracy'] = surf
        plt.close(fig_accuracy)
        
        # Create d-prime history graph
        fig_dprime, ax_dprime = plt.subplots(figsize=(6, 4), dpi=80)
        
        # Aggregate d-prime data by date
        date_pos_dprimes = []
        date_letter_dprimes = []
        
        for date in unique_dates:
            indices = date_grouped_indices[date]
            date_pos_dprimes.append(np.mean([position_dprimes[i] for i in indices]))
            date_letter_dprimes.append(np.mean([letter_dprimes[i] for i in indices]))
        
        # Plot daily data points
        ax_dprime.plot(x_points, date_pos_dprimes, 'b-o', label='Position d\'')
        ax_dprime.plot(x_points, date_letter_dprimes, 'g-o', label='Letter d\'')
        
        # Add moving averages if we have enough data points
        if len(date_pos_dprimes) >= window_size:
            ma_x = x_points[window_size-1:]
            pos_dprime_ma = moving_average(date_pos_dprimes, window_size)
            letter_dprime_ma = moving_average(date_letter_dprimes, window_size)
            
            ax_dprime.plot(ma_x, pos_dprime_ma, 'b--', alpha=0.7, linewidth=2, label='Position MA')
            ax_dprime.plot(ma_x, letter_dprime_ma, 'g--', alpha=0.7, linewidth=2, label='Letter MA')
        
        ax_dprime.set_title('D-Prime Progress Over Time')
        ax_dprime.set_xlabel('Date')
        ax_dprime.set_ylabel('d-prime')
        ax_dprime.grid(True, linestyle='--', alpha=0.7)
        ax_dprime.legend()
        
        # Format x-axis with dates
        ax_dprime.set_xticks(x_points)
        ax_dprime.set_xticklabels(unique_dates, rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_dprime)
        canvas.draw()
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['history_dprime'] = surf
        plt.close(fig_dprime)
        
        # Create n-level history graph
        fig_nlevel, ax_nlevel = plt.subplots(figsize=(6, 4), dpi=80)
        
        # Aggregate n-level data by date
        date_n_levels = []
        
        for date in unique_dates:
            indices = date_grouped_indices[date]
            date_n_levels.append(np.mean([n_levels[i] for i in indices]))
        
        # Plot daily data points
        ax_nlevel.plot(x_points, date_n_levels, 'r-o', label='N-Level')
        
        # Add moving average if we have enough data points
        if len(date_n_levels) >= window_size:
            ma_x = x_points[window_size-1:]
            n_level_ma = moving_average(date_n_levels, window_size)
            
            ax_nlevel.plot(ma_x, n_level_ma, 'r--', alpha=0.7, linewidth=2, label='N-Level MA')
        
        ax_nlevel.set_title('N-Level Progress Over Time')
        ax_nlevel.set_xlabel('Date')
        ax_nlevel.set_ylabel('N-Level')
        ax_nlevel.set_ylim(0, MAX_N + 1)
        ax_nlevel.grid(True, linestyle='--', alpha=0.7)
        ax_nlevel.legend()
        
        # Format x-axis with dates
        ax_nlevel.set_xticks(x_points)
        ax_nlevel.set_xticklabels(unique_dates, rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to pygame surface
        canvas = agg.FigureCanvasAgg(fig_nlevel)
        canvas.draw()
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.graph_surfaces['history_nlevel'] = surf
        plt.close(fig_nlevel)
        
        return True
    
    def draw_grid(self):
        grid_width = GRID_SIZE * self.cell_size
        grid_height = GRID_SIZE * self.cell_size
        grid_left = (self.window_width - grid_width) // 2
        grid_top = (self.window_height - grid_height) // 2 - 50  # Offset to make room for UI elements
        
        # Draw grid background
        pygame.draw.rect(self.screen, GRAY, (grid_left - 5, grid_top - 5, 
                                           grid_width + 10, grid_height + 10))
        
        # Draw cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_idx = row * GRID_SIZE + col
                x = grid_left + col * self.cell_size
                y = grid_top + row * self.cell_size
                
                cell_color = WHITE
                
                # Highlight current position
                if cell_idx == self.current_position:
                    cell_color = BLUE
                    
                    # Draw the letter in the center of the active square
                    if self.current_displayed_letter is not None and self.visual_only_mode:
                        letter_font = pygame.font.SysFont(None, int(self.cell_size * 0.7))
                        letter_surf = letter_font.render(self.current_displayed_letter, True, BLACK)
                        letter_pos = (x + self.cell_size // 2 - letter_surf.get_width() // 2,
                                    y + self.cell_size // 2 - letter_surf.get_height() // 2)
                        self.screen.blit(letter_surf, letter_pos)
                
                pygame.draw.rect(self.screen, cell_color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, BLACK, (x, y, self.cell_size, self.cell_size), 2)
                
                # Draw the letter in the center of the active square (after drawing the cell)
                if cell_idx == self.current_position and self.current_displayed_letter is not None and self.visual_only_mode:
                    letter_font = pygame.font.SysFont(None, int(self.cell_size * 0.7))
                    letter_surf = letter_font.render(self.current_displayed_letter, True, BLACK)
                    letter_pos = (x + self.cell_size // 2 - letter_surf.get_width() // 2,
                                y + self.cell_size // 2 - letter_surf.get_height() // 2)
                    self.screen.blit(letter_surf, letter_pos)
        
        return grid_left, grid_top, grid_width, grid_height
    
    def draw_session_info(self):
        # Draw N-back level
        n_text = self.font.render(f"N-Level: {self.n_level}", True, WHITE)
        self.screen.blit(n_text, (20, 20))
        
        # Draw progress
        progress_text = self.font.render(f"Block: {self.current_block + 1}/{self.blocks_per_session}  Trial: {self.current_trial + 1}/{self.trials_per_block}", True, WHITE)
        self.screen.blit(progress_text, (self.window_width - 20 - progress_text.get_width(), 20))
        
        # Draw current sound/letter if not in visual-only mode
        if self.current_displayed_letter is not None and not self.visual_only_mode:
            # Draw letter in a highlighted box
            letter_box = pygame.Rect(self.window_width // 2 - 25, 20, 50, 50)
            pygame.draw.rect(self.screen, YELLOW, letter_box)
            
            sound_text = self.font.render(self.current_displayed_letter, True, BLACK)
            self.screen.blit(sound_text, (letter_box.centerx - sound_text.get_width() // 2, 
                                       letter_box.centery - sound_text.get_height() // 2))
                                       
        # Display real-time scores if enabled
        if self.show_realtime_score and self.current_trial > 0:
            # Calculate current accuracy
            scores_y_pos = 70
            
            # Calculate position accuracy and d-prime
            if hasattr(self, 'position_correct') and hasattr(self, 'position_false_positive') and hasattr(self, 'position_false_negative'):
                total_position_targets = self.position_correct + self.position_false_negative
                position_accuracy = 0
                position_dprime = 0
                
                if total_position_targets > 0:
                    position_accuracy = self.position_correct / total_position_targets
                    
                    # Calculate position d-prime if we have enough data
                    if total_position_targets > 0 and self.current_trial - total_position_targets > 0:
                        # Calculate hit rate and false alarm rate
                        position_hit_rate = self.position_correct / (total_position_targets + 0.01)  # Add small constant to avoid division by zero
                        position_fa_rate = self.position_false_positive / ((self.current_trial - total_position_targets) + 0.01)
                        # Calculate d-prime
                        position_dprime = self.calculate_dprime(position_hit_rate, position_fa_rate)
                    
                    # Display position accuracy and d-prime
                    pos_acc_text = self.font.render(f"Position Accuracy: {position_accuracy:.2f}", True, GREEN)
                    self.screen.blit(pos_acc_text, (20, scores_y_pos))
                    scores_y_pos += 30
                    
                    pos_dprime_text = self.font.render(f"Position D-Prime: {position_dprime:.2f}", True, GREEN)
                    self.screen.blit(pos_dprime_text, (20, scores_y_pos))
                    scores_y_pos += 30
                
                # Calculate sound/letter accuracy and d-prime
                total_sound_targets = self.sound_correct + self.sound_false_negative
                sound_accuracy = 0
                sound_dprime = 0
                
                if total_sound_targets > 0:
                    sound_accuracy = self.sound_correct / total_sound_targets
                    
                    # Calculate sound/letter d-prime if we have enough data
                    if total_sound_targets > 0 and self.current_trial - total_sound_targets > 0:
                        # Calculate hit rate and false alarm rate
                        sound_hit_rate = self.sound_correct / (total_sound_targets + 0.01)
                        sound_fa_rate = self.sound_false_positive / ((self.current_trial - total_sound_targets) + 0.01)
                        # Calculate d-prime
                        sound_dprime = self.calculate_dprime(sound_hit_rate, sound_fa_rate)
                    
                    # Display sound/letter accuracy and d-prime
                    sound_acc_text = self.font.render(f"Letter Accuracy: {sound_accuracy:.2f}", True, GREEN)
                    self.screen.blit(sound_acc_text, (20, scores_y_pos))
                    scores_y_pos += 30
                    
                    sound_dprime_text = self.font.render(f"Letter D-Prime: {sound_dprime:.2f}", True, GREEN)
                    self.screen.blit(sound_dprime_text, (20, scores_y_pos))
    
    def draw_controls(self, grid_left, grid_top, grid_width, grid_height):
        # Position match control
        position_button_rect = pygame.Rect(grid_left, grid_top + grid_height + 20, 
                                         grid_width // 2 - 10, 40)
        pygame.draw.rect(self.screen, BLUE, position_button_rect, 0, 5)
        
        position_text = self.font.render("Position Match (A)", True, WHITE)
        text_x = position_button_rect.centerx - position_text.get_width() // 2
        text_y = position_button_rect.centery - position_text.get_height() // 2
        self.screen.blit(position_text, (text_x, text_y))
        
        # Sound match control
        sound_button_rect = pygame.Rect(grid_left + grid_width // 2 + 10, grid_top + grid_height + 20,
                                      grid_width // 2 - 10, 40)
        pygame.draw.rect(self.screen, YELLOW, sound_button_rect, 0, 5)
        
        sound_text = self.font.render("Letter Match (L)", True, BLACK)  # Changed "Sound" to "Letter" for clarity
        text_x = sound_button_rect.centerx - sound_text.get_width() // 2
        text_y = sound_button_rect.centery - sound_text.get_height() // 2
        self.screen.blit(sound_text, (text_x, text_y))
        
        # Draw feedback if user responded or missed a match
        
        # Position feedback
        if self.position_response is not None:
            # User made a response
            feedback_color = GREEN if (self.position_response and self.position_match) or \
                                     (not self.position_response and not self.position_match) else RED
            pygame.draw.rect(self.screen, feedback_color, position_button_rect, 3, 5)
        elif self.missed_position_match:
            # User missed a position match
            pygame.draw.rect(self.screen, RED, position_button_rect, 3, 5)
            
            # Add a missed match indicator
            miss_text = self.font.render("MISSED!", True, RED)
            miss_x = position_button_rect.centerx - miss_text.get_width() // 2
            miss_y = position_button_rect.bottom + 10
            self.screen.blit(miss_text, (miss_x, miss_y))
        
        # Sound/Letter feedback
        if self.sound_response is not None:
            # User made a response
            feedback_color = GREEN if (self.sound_response and self.sound_match) or \
                                     (not self.sound_response and not self.sound_match) else RED
            pygame.draw.rect(self.screen, feedback_color, sound_button_rect, 3, 5)
        elif self.missed_sound_match:
            # User missed a sound/letter match
            pygame.draw.rect(self.screen, RED, sound_button_rect, 3, 5)
            
            # Add a missed match indicator
            miss_text = self.font.render("MISSED!", True, RED)
            miss_x = sound_button_rect.centerx - miss_text.get_width() // 2
            miss_y = sound_button_rect.bottom + 10
            self.screen.blit(miss_text, (miss_x, miss_y))
    
    def draw_menu(self):
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.font.render("Dual N-Back Memory Training", True, WHITE)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 100))
        
        # Draw user stats if available
        if self.user_data['total_trials'] > 0:
            stats_text = [
                f"Highest N-Level: {self.user_data['highest_n_level']}",
                f"Total Trials: {self.user_data['total_trials']}",
                f"Average Position Accuracy: {self.user_data['average_accuracy']['position']:.2f}",
                f"Average Sound Accuracy: {self.user_data['average_accuracy']['sound']:.2f}"
            ]
            
            for i, text in enumerate(stats_text):
                stat_surf = self.small_font.render(text, True, WHITE)
                self.screen.blit(stat_surf, (self.window_width // 2 - stat_surf.get_width() // 2, 180 + i * 30))
        
        # Calculate button positions
        button_width = 200
        button_height = 50
        button_y_start = 350
        button_spacing = 20
        
        # Draw start button
        start_button = pygame.Rect(self.window_width // 2 - button_width // 2, button_y_start, button_width, button_height)
        pygame.draw.rect(self.screen, BLUE, start_button, 0, 10)
        
        start_text = self.font.render("Start Session", True, WHITE)
        self.screen.blit(start_text, (self.window_width // 2 - start_text.get_width() // 2, 
                                   start_button.centery - start_text.get_height() // 2))
        
        # Draw history button
        history_button = pygame.Rect(self.window_width // 2 - button_width // 2, 
                               button_y_start + button_height + button_spacing, 
                               button_width, button_height)
        pygame.draw.rect(self.screen, GREEN, history_button, 0, 10)
        
        history_text = self.font.render("Progress History", True, BLACK)
        self.screen.blit(history_text, (self.window_width // 2 - history_text.get_width() // 2, 
                                     history_button.centery - history_text.get_height() // 2))
        
        # Draw settings button
        settings_button = pygame.Rect(self.window_width // 2 - button_width // 2, 
                                 history_button.bottom + button_spacing, 
                                 button_width, button_height)
        pygame.draw.rect(self.screen, GRAY, settings_button, 0, 10)
        
        settings_text = self.font.render("Settings", True, WHITE)
        self.screen.blit(settings_text, (self.window_width // 2 - settings_text.get_width() // 2, 
                                      settings_button.centery - settings_text.get_height() // 2))
        
        # Draw quit button
        quit_button = pygame.Rect(self.window_width // 2 - button_width // 2, 
                             settings_button.bottom + button_spacing, 
                             button_width, button_height)
        pygame.draw.rect(self.screen, RED, quit_button, 0, 10)
        
        quit_text = self.font.render("Quit", True, WHITE)
        self.screen.blit(quit_text, (self.window_width // 2 - quit_text.get_width() // 2, 
                                  quit_button.centery - quit_text.get_height() // 2))
        
        # Add keyboard shortcut info
        keys_text = self.small_font.render("Press 'A' for position match, 'L' for letter match, Space for both", True, WHITE)
        self.screen.blit(keys_text, (self.window_width // 2 - keys_text.get_width() // 2, self.window_height - 40))
        
        # Return the button rects for click detection
        return start_button, history_button, settings_button, quit_button
        
    def draw_history(self):
        """Draw the history screen with progress graphs"""
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.font.render("Training Progress History", True, WHITE)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 20))
        
        # Check if we have history data
        if not hasattr(self, 'graph_surfaces') or not self.graph_surfaces or 'history_accuracy' not in self.graph_surfaces:
            has_graphs = self.generate_history_graphs()
            if not has_graphs:
                # No history data available
                no_data_text = self.font.render("No training history available yet.", True, WHITE)
                self.screen.blit(no_data_text, (self.window_width // 2 - no_data_text.get_width() // 2, 
                                            self.window_height // 2 - no_data_text.get_height() // 2))
                
                # Draw back button
                back_button = pygame.Rect(self.window_width // 2 - 100, self.window_height - 100, 200, 50)
                pygame.draw.rect(self.screen, RED, back_button, 0, 10)
                
                back_text = self.font.render("Back to Menu", True, WHITE)
                self.screen.blit(back_text, (self.window_width // 2 - back_text.get_width() // 2,
                                         back_button.centery - back_text.get_height() // 2))
                
                return (back_button,), 0  # Return back button and page count
        
        # Determine which graphs to show based on page
        graphs_y = 70
        graph_spacing = 20
        
        # Page 0: Show accuracy and n-level
        if self.history_page == 0:
            # Accuracy graph
            if 'history_accuracy' in self.graph_surfaces:
                acc_graph = self.graph_surfaces['history_accuracy']
                self.screen.blit(acc_graph, (self.window_width // 2 - acc_graph.get_width() // 2, graphs_y))
                graphs_y += acc_graph.get_height() + graph_spacing
            
            # N-level graph
            if 'history_nlevel' in self.graph_surfaces:
                nlevel_graph = self.graph_surfaces['history_nlevel']
                self.screen.blit(nlevel_graph, (self.window_width // 2 - nlevel_graph.get_width() // 2, graphs_y))
                graphs_y += nlevel_graph.get_height() + graph_spacing
                
        # Page 1: Show d-prime
        elif self.history_page == 1:
            # D-prime graph
            if 'history_dprime' in self.graph_surfaces:
                dprime_graph = self.graph_surfaces['history_dprime']
                self.screen.blit(dprime_graph, (self.window_width // 2 - dprime_graph.get_width() // 2, graphs_y))
                graphs_y += dprime_graph.get_height() + graph_spacing
        
        # Navigation buttons at the bottom
        button_width = 150
        button_height = 50
        button_spacing = 20
        buttons_y = self.window_height - 120  # Moved up to make room for extra info

        # Info about the graph view
        info_text_lines = [
            "• Data points are averaged by day",
            "• Dashed lines show moving averages",
            "• Equal spacing between days on x-axis"
        ]
        
        for i, text in enumerate(info_text_lines):
            info_surf = self.small_font.render(text, True, GRAY)
            self.screen.blit(info_surf, (20, buttons_y - 70 + i * 20))
        
        # Previous page button (if not on first page)
        if self.history_page > 0:
            prev_button = pygame.Rect(self.window_width // 4 - button_width // 2, buttons_y, button_width, button_height)
            pygame.draw.rect(self.screen, BLUE, prev_button, 0, 10)
            
            prev_text = self.font.render("Previous", True, WHITE)
            self.screen.blit(prev_text, (prev_button.centerx - prev_text.get_width() // 2,
                                     prev_button.centery - prev_text.get_height() // 2))
        else:
            prev_button = None
        
        # Next page button (if not on last page)
        total_pages = 2  # Currently we have 2 pages of graphs
        if self.history_page < total_pages - 1:
            next_button = pygame.Rect(self.window_width * 3 // 4 - button_width // 2, buttons_y, button_width, button_height)
            pygame.draw.rect(self.screen, BLUE, next_button, 0, 10)
            
            next_text = self.font.render("Next", True, WHITE)
            self.screen.blit(next_text, (next_button.centerx - next_text.get_width() // 2,
                                     next_button.centery - next_text.get_height() // 2))
        else:
            next_button = None
        
        # Back button
        back_button = pygame.Rect(self.window_width // 2 - button_width // 2, buttons_y, button_width, button_height)
        pygame.draw.rect(self.screen, RED, back_button, 0, 10)
        
        back_text = self.font.render("Back to Menu", True, WHITE)
        self.screen.blit(back_text, (back_button.centerx - back_text.get_width() // 2,
                                 back_button.centery - back_text.get_height() // 2))
        
        # Page indicator
        page_text = self.small_font.render(f"Page {self.history_page + 1} of {total_pages}", True, WHITE)
        self.screen.blit(page_text, (self.window_width // 2 - page_text.get_width() // 2, 
                                 buttons_y - page_text.get_height() - 10))
        
        # Return all buttons and total page count
        buttons = []
        if prev_button:
            buttons.append(prev_button)
        buttons.append(back_button)
        if next_button:
            buttons.append(next_button)
            
        return tuple(buttons), total_pages
    
    def draw_settings(self):
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.font.render("Settings", True, WHITE)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 20))
        
        # Define margins and spacing
        left_margin = 20
        right_margin = 20
        top_margin = 70
        
        # Game Settings Section - Left Column
        settings_col1_x = left_margin
        col1_width = (self.window_width // 2) - 30
        
        # Draw column divider line and section header
        pygame.draw.line(self.screen, GRAY, (self.window_width // 2, top_margin), 
                        (self.window_width // 2, self.window_height - 80), 2)
        
        settings_title = self.font.render("Game Settings", True, YELLOW)
        self.screen.blit(settings_title, (settings_col1_x, top_margin))
        
        y_pos = top_margin + 40
        
        # Toggle settings
        # Each setting will be placed side by side
        toggle_width = 80
        toggle_height = 40
        toggle_spacing = 20
        
        # Visual mode toggle
        visual_label = self.font.render("Visual Mode:", True, WHITE)
        self.screen.blit(visual_label, (settings_col1_x, y_pos))
        
        visual_toggle_rect = pygame.Rect(settings_col1_x + 170, y_pos, toggle_width, toggle_height)
        visual_color = GREEN if self.visual_only_mode else RED
        pygame.draw.rect(self.screen, visual_color, visual_toggle_rect, 0, 5)
        
        toggle_text = self.font.render("ON" if self.visual_only_mode else "OFF", True, WHITE)
        self.screen.blit(toggle_text, (visual_toggle_rect.centerx - toggle_text.get_width() // 2,
                                  visual_toggle_rect.centery - toggle_text.get_height() // 2))
                                  
        # Real-time score toggle
        y_pos += toggle_height + 15
        score_label = self.font.render("Show Scores:", True, WHITE)
        self.screen.blit(score_label, (settings_col1_x, y_pos))
        
        score_toggle_rect = pygame.Rect(settings_col1_x + 170, y_pos, toggle_width, toggle_height)
        score_color = GREEN if self.show_realtime_score else RED
        pygame.draw.rect(self.screen, score_color, score_toggle_rect, 0, 5)
        
        toggle_text = self.font.render("ON" if self.show_realtime_score else "OFF", True, WHITE)
        self.screen.blit(toggle_text, (score_toggle_rect.centerx - toggle_text.get_width() // 2,
                                  score_toggle_rect.centery - toggle_text.get_height() // 2))
        
        # Value adjustment settings
        y_pos += toggle_height + 30
        button_size = (40, 40)
        
        # Trial duration
        duration_label = self.font.render(f"Trial Duration: {self.trial_duration}s", True, WHITE)
        self.screen.blit(duration_label, (settings_col1_x, y_pos))
        
        duration_desc = self.small_font.render(f"({MIN_TRIAL_DURATION}s - {MAX_TRIAL_DURATION}s)", True, GRAY)
        self.screen.blit(duration_desc, (settings_col1_x, y_pos + 30))
        
        decrease_x = settings_col1_x + 240
        duration_decrease = pygame.Rect(decrease_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, duration_decrease, 0, 5)
        self.screen.blit(self.font.render("-", True, WHITE), (duration_decrease.centerx - 5, duration_decrease.centery - 10))
        
        increase_x = decrease_x + button_size[0] + 5
        duration_increase = pygame.Rect(increase_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, duration_increase, 0, 5)
        self.screen.blit(self.font.render("+", True, WHITE), (duration_increase.centerx - 5, duration_increase.centery - 10))
        
        # Trials per block
        y_pos += 60
        trials_label = self.font.render(f"Trials per Block: {self.trials_per_block}", True, WHITE)
        self.screen.blit(trials_label, (settings_col1_x, y_pos))
        
        trials_desc = self.small_font.render(f"({MIN_TRIALS_PER_BLOCK} - {MAX_TRIALS_PER_BLOCK})", True, GRAY)
        self.screen.blit(trials_desc, (settings_col1_x, y_pos + 30))
        
        trials_decrease = pygame.Rect(decrease_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, trials_decrease, 0, 5)
        self.screen.blit(self.font.render("-", True, WHITE), (trials_decrease.centerx - 5, trials_decrease.centery - 10))
        
        trials_increase = pygame.Rect(increase_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, trials_increase, 0, 5)
        self.screen.blit(self.font.render("+", True, WHITE), (trials_increase.centerx - 5, trials_increase.centery - 10))
        
        # Blocks per session
        y_pos += 60
        blocks_label = self.font.render(f"Blocks per Session: {self.blocks_per_session}", True, WHITE)
        self.screen.blit(blocks_label, (settings_col1_x, y_pos))
        
        blocks_desc = self.small_font.render(f"({MIN_BLOCKS_PER_SESSION} - {MAX_BLOCKS_PER_SESSION})", True, GRAY)
        self.screen.blit(blocks_desc, (settings_col1_x, y_pos + 30))
        
        blocks_decrease = pygame.Rect(decrease_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, blocks_decrease, 0, 5)
        self.screen.blit(self.font.render("-", True, WHITE), (blocks_decrease.centerx - 5, blocks_decrease.centery - 10))
        
        blocks_increase = pygame.Rect(increase_x, y_pos, button_size[0], button_size[1])
        pygame.draw.rect(self.screen, BLUE, blocks_increase, 0, 5)
        self.screen.blit(self.font.render("+", True, WHITE), (blocks_increase.centerx - 5, blocks_increase.centery - 10))
        
        # Information Section - Right Column
        settings_col2_x = (self.window_width // 2) + 15
        col2_width = (self.window_width // 2) - 30
        
        # N-level adjustment rules explanation
        rules_title = self.font.render("N-Level Adjustment Rules", True, YELLOW)
        self.screen.blit(rules_title, (settings_col2_x, top_margin))
        
        rules_y = top_margin + 40
        rules_spacing = 25
        
        rules_text = [
            "• N-level increases if both accuracies are ≥ 80%",
            "  AND both d-primes are ≥ 2.0",
            "• N-level decreases if either accuracy is ≤ 50%",
            "  OR both d-primes are < 1.0",
            "• N-level stays the same if performance is between",
            "  these thresholds",
            f"• Minimum N-level: {MIN_N}, Maximum N-level: {MAX_N}"
        ]
        
        for rule in rules_text:
            rule_text = self.small_font.render(rule, True, WHITE)
            self.screen.blit(rule_text, (settings_col2_x, rules_y))
            rules_y += 22
        
        # D-prime explanation
        dprime_y = rules_y + 15
        dprime_title = self.font.render("D-Prime Explanation", True, YELLOW)
        self.screen.blit(dprime_title, (settings_col2_x, dprime_y))
        
        dprime_y += 35
        dprime_text = [
            "D-prime measures sensitivity between matches",
            "and non-matches:",
            "",
            "• Hit Rate = Correct matches / Total matches",
            "• False Alarm = Wrong responses / Non-match trials",
            "• D-prime = Z(Hit Rate) - Z(False Alarm Rate)",
            "",
            "• < 1.0: Poor discrimination",
            "• 1.0-2.0: Moderate discrimination",
            "• > 2.0: Excellent discrimination"
        ]
        
        for line in dprime_text:
            line_color = GRAY if line.startswith("•") else WHITE
            line_text = self.small_font.render(line, True, line_color)
            self.screen.blit(line_text, (settings_col2_x, dprime_y))
            dprime_y += 22
        
        # Back button at the bottom center
        back_button_width = 200
        back_button_height = 50
        back_button = pygame.Rect(self.window_width // 2 - back_button_width // 2, 
                              self.window_height - 70, 
                              back_button_width, back_button_height)
        pygame.draw.rect(self.screen, RED, back_button, 0, 10)
        
        back_text = self.font.render("Back to Menu", True, WHITE)
        self.screen.blit(back_text, (back_button.centerx - back_text.get_width() // 2,
                                   back_button.centery - back_text.get_height() // 2))
        
        # Add keyboard shortcut info at the bottom
        shortcuts = "ESC: Menu   V: Toggle Visual   R: Toggle Scores"
        shortcut_text = self.small_font.render(shortcuts, True, WHITE)
        shortcut_x = self.window_width // 2 - shortcut_text.get_width() // 2
        self.screen.blit(shortcut_text, (shortcut_x, self.window_height - 20))
        
        # Return button rects for click detection
        return visual_toggle_rect, score_toggle_rect, duration_decrease, duration_increase, trials_decrease, trials_increase, blocks_decrease, blocks_increase, back_button
    
    def draw_block_summary(self):
        self.screen.fill(BLACK)
        
        # Get the most recent block results
        block = self.block_results[-1]
        
        # Draw title
        title_text = self.font.render(f"Block {self.current_block} Summary", True, WHITE)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 80))
        
        # Draw performance metrics
        metrics = [
            f"N-Level: {block['n_level']}",
            f"Position Accuracy: {block['position_accuracy']:.2f}",
            f"Sound Accuracy: {block['sound_accuracy']:.2f}",
            f"Position D-Prime: {block['position_dprime']:.2f}",
            f"Sound D-Prime: {block['sound_dprime']:.2f}"
        ]
        
        for i, text in enumerate(metrics):
            metric_surf = self.font.render(text, True, WHITE)
            self.screen.blit(metric_surf, (self.window_width // 2 - metric_surf.get_width() // 2, 130 + i * 30))
        
        # Show N-level change explanation
        n_level_change_y = 300
        position_accuracy = block['position_accuracy']
        sound_accuracy = block['sound_accuracy']
        position_dprime = block['position_dprime']
        sound_dprime = block['sound_dprime']
        good_dprime = position_dprime >= 2.0 and sound_dprime >= 2.0
        
        if position_accuracy >= 0.8 and sound_accuracy >= 0.8 and good_dprime and block['n_level'] < MAX_N:
            explanation = "N-level will increase for the next block (good accuracy & d-prime scores)"
            color = GREEN
        elif ((position_accuracy <= 0.5 or sound_accuracy <= 0.5) or 
              (position_dprime < 1.0 and sound_dprime < 1.0)) and block['n_level'] > MIN_N:
            if position_accuracy <= 0.5 or sound_accuracy <= 0.5:
                explanation = "N-level will decrease for the next block (low accuracy scores)"
            else:
                explanation = "N-level will decrease for the next block (low d-prime scores)"
            color = RED
        else:
            explanation = "N-level will remain the same for the next block"
            color = YELLOW
            
        explanation_text = self.font.render(explanation, True, color)
        self.screen.blit(explanation_text, (self.window_width // 2 - explanation_text.get_width() // 2, n_level_change_y))
        
        # Draw continue button
        if self.current_block < self.blocks_per_session:
            next_text = "Continue to Next Block"
        else:
            next_text = "View Session Summary"
        
        continue_button = pygame.Rect(self.window_width // 2 - 150, 420, 300, 50)
        pygame.draw.rect(self.screen, BLUE, continue_button, 0, 10)
        
        continue_text = self.font.render(next_text, True, WHITE)
        self.screen.blit(continue_text, (self.window_width // 2 - continue_text.get_width() // 2, 
                                      continue_button.centery - continue_text.get_height() // 2))
        
        # Add keyboard shortcut info
        keys_text = self.small_font.render("Press ENTER or SPACE to continue", True, WHITE)
        self.screen.blit(keys_text, (self.window_width // 2 - keys_text.get_width() // 2, self.window_height - 40))
        
        return continue_button
    
    def draw_session_summary(self):
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.font.render("Session Complete!", True, WHITE)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 30))
        
        # Draw session metrics
        metrics_y = 70
        metrics = [
            f"Highest N-Level: {self.session_results['highest_n_level']}",
            f"Average Position Accuracy: {self.session_results['avg_position_accuracy']:.2f}",
            f"Average Letter Accuracy: {self.session_results['avg_sound_accuracy']:.2f}",
            f"Average Position D-Prime: {self.session_results['position_dprime']:.2f}",
            f"Average Letter D-Prime: {self.session_results['sound_dprime']:.2f}"
        ]
        
        for i, text in enumerate(metrics):
            metric_surf = self.font.render(text, True, WHITE)
            self.screen.blit(metric_surf, (self.window_width // 2 - metric_surf.get_width() // 2, metrics_y + i * 30))
        
        # Draw performance graphs if available
        graphs_y = 230
        
        if 'accuracy' in self.graph_surfaces:
            # Draw accuracy graph
            acc_graph = self.graph_surfaces['accuracy']
            self.screen.blit(acc_graph, (self.window_width // 2 - acc_graph.get_width() // 2, graphs_y))
            graphs_y += acc_graph.get_height() + 20
        
        if 'dprime' in self.graph_surfaces:
            # Draw d-prime graph
            dprime_graph = self.graph_surfaces['dprime']
            self.screen.blit(dprime_graph, (self.window_width // 2 - dprime_graph.get_width() // 2, graphs_y))
            graphs_y += dprime_graph.get_height() + 20
        
        if 'n_level' in self.graph_surfaces:
            # Draw N-level progression graph
            nlevel_graph = self.graph_surfaces['n_level']
            self.screen.blit(nlevel_graph, (self.window_width // 2 - nlevel_graph.get_width() // 2, graphs_y))
            graphs_y += nlevel_graph.get_height() + 20
        
        # If no graphs available, show basic chart
        if not self.graph_surfaces and len(self.block_results) > 0:
            # Draw N-level progress over blocks as simple bar chart
            progress_title = self.font.render("N-Level Progress", True, WHITE)
            self.screen.blit(progress_title, (self.window_width // 2 - progress_title.get_width() // 2, graphs_y))
            
            n_levels = [block['n_level'] for block in self.block_results]
            max_blocks_to_show = min(self.blocks_per_session, 20)  # Show at most 20 blocks for clarity
            if len(n_levels) > max_blocks_to_show:
                n_levels = n_levels[-max_blocks_to_show:]
            
            # Draw simple bar chart
            bar_width = min(20, (self.window_width - 100) // max(1, len(n_levels)))
            chart_width = bar_width * len(n_levels)
            chart_left = (self.window_width - chart_width) // 2
            chart_y = graphs_y + 40
            
            for i, n in enumerate(n_levels):
                bar_height = n * 15  # Scale height by N level
                bar_x = chart_left + i * bar_width
                bar_y = chart_y + 100 - bar_height
                pygame.draw.rect(self.screen, BLUE, (bar_x, bar_y, bar_width - 2, bar_height))
                
                # Draw block number
                if i % 5 == 0 or i == len(n_levels) - 1:  # Show every 5th block number
                    block_num = i + 1
                    num_surf = self.small_font.render(str(block_num), True, WHITE)
                    self.screen.blit(num_surf, (bar_x, chart_y + 105))
            
            graphs_y = chart_y + 140  # Update y position for next element
        
        # Draw return to menu button
        menu_button = pygame.Rect(self.window_width // 2 - 100, self.window_height - 80, 200, 50)
        pygame.draw.rect(self.screen, GREEN, menu_button, 0, 10)
        
        menu_text = self.font.render("Return to Menu", True, BLACK)
        self.screen.blit(menu_text, (self.window_width // 2 - menu_text.get_width() // 2, 
                                   menu_button.centery - menu_text.get_height() // 2))
        
        # Add keyboard shortcut info
        keys_text = self.small_font.render("Press ENTER, SPACE, or ESC to return to menu", True, WHITE)
        self.screen.blit(keys_text, (self.window_width // 2 - keys_text.get_width() // 2, self.window_height - 20))
        
        return menu_button
    
    def draw_pause_screen(self):
        """Draw the pause screen overlay"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Draw pause title
        pause_text = self.font.render("Game Paused", True, WHITE)
        self.screen.blit(pause_text, (self.window_width // 2 - pause_text.get_width() // 2, 
                                  self.window_height // 3))
        
        # Draw buttons
        button_width = 200
        button_height = 50
        button_spacing = 20
        buttons_y = self.window_height // 2
        
        # Resume button
        resume_button = pygame.Rect(self.window_width // 2 - button_width // 2, 
                               buttons_y, 
                               button_width, button_height)
        pygame.draw.rect(self.screen, GREEN, resume_button, 0, 10)
        
        resume_text = self.font.render("Resume Game", True, BLACK)
        self.screen.blit(resume_text, (resume_button.centerx - resume_text.get_width() // 2, 
                                   resume_button.centery - resume_text.get_height() // 2))
        
        # Return to menu button
        menu_button = pygame.Rect(self.window_width // 2 - button_width // 2, 
                             buttons_y + button_height + button_spacing, 
                             button_width, button_height)
        pygame.draw.rect(self.screen, BLUE, menu_button, 0, 10)
        
        menu_text = self.font.render("Return to Menu", True, WHITE)
        self.screen.blit(menu_text, (menu_button.centerx - menu_text.get_width() // 2, 
                                 menu_button.centery - menu_text.get_height() // 2))
        
        # Instructions
        instructions = self.small_font.render("Press P to resume or ESC to return to menu", True, WHITE)
        self.screen.blit(instructions, (self.window_width // 2 - instructions.get_width() // 2, 
                                    menu_button.bottom + 30))
        
        return resume_button, menu_button
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            # Handle window resize events
            if event.type == pygame.VIDEORESIZE:
                self.window_width, self.window_height = event.size
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                # Adjust cell size based on new window dimensions
                grid_area = min(self.window_width * 0.8, self.window_height * 0.6)
                self.cell_size = int(grid_area / GRID_SIZE)
            
            # Global pause key for playing state
            if self.game_state == "playing" and event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.paused = not self.paused
                return
            
            # Handle pause screen events
            if self.game_state == "playing" and self.paused:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    resume_button, menu_button = self.draw_pause_screen()
                    
                    if resume_button.collidepoint(mouse_pos):
                        self.paused = False
                    elif menu_button.collidepoint(mouse_pos):
                        self.paused = False
                        self.game_state = "menu"
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p or event.key == pygame.K_SPACE:
                        self.paused = False
                    elif event.key == pygame.K_ESCAPE:
                        self.paused = False
                        self.game_state = "menu"
                return
            
            if self.game_state == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    start_button, history_button, settings_button, quit_button = self.draw_menu()
                    
                    if start_button.collidepoint(mouse_pos):
                        self.start_session()
                    elif history_button.collidepoint(mouse_pos):
                        self.history_page = 0  # Reset to first page
                        self.game_state = "history"
                    elif settings_button.collidepoint(mouse_pos):
                        self.game_state = "settings"
                    elif quit_button.collidepoint(mouse_pos):
                        self.running = False
                        return
                
                # Add keyboard shortcuts for menu navigation
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Enter key to start
                        self.start_session()
                    elif event.key == pygame.K_h:  # H key for history
                        self.history_page = 0
                        self.game_state = "history"
                    elif event.key == pygame.K_s:  # S key for settings
                        self.game_state = "settings"
                    elif event.key == pygame.K_ESCAPE:  # Escape to quit
                        self.running = False
                        return
            
            elif self.game_state == "settings":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    visual_toggle, score_toggle, duration_decrease, duration_increase, trials_decrease, trials_increase, blocks_decrease, blocks_increase, back_button = self.draw_settings()
                    
                    if visual_toggle.collidepoint(mouse_pos):
                        # Toggle visual-only mode
                        self.visual_only_mode = not self.visual_only_mode
                    elif score_toggle.collidepoint(mouse_pos):
                        # Toggle realtime score display
                        self.show_realtime_score = not self.show_realtime_score
                    elif duration_decrease.collidepoint(mouse_pos):
                        # Decrease trial duration (with min limit)
                        self.trial_duration = max(MIN_TRIAL_DURATION, self.trial_duration - 0.5)
                    elif duration_increase.collidepoint(mouse_pos):
                        # Increase trial duration (with max limit)
                        self.trial_duration = min(MAX_TRIAL_DURATION, self.trial_duration + 0.5)
                    elif trials_decrease.collidepoint(mouse_pos):
                        # Decrease trials per block (with min limit)
                        self.trials_per_block = max(MIN_TRIALS_PER_BLOCK, self.trials_per_block - 5)
                    elif trials_increase.collidepoint(mouse_pos):
                        # Increase trials per block (with max limit)
                        self.trials_per_block = min(MAX_TRIALS_PER_BLOCK, self.trials_per_block + 5)
                    elif blocks_decrease.collidepoint(mouse_pos):
                        # Decrease blocks per session (with min lfimit)
                        self.blocks_per_session = max(MIN_BLOCKS_PER_SESSION, self.blocks_per_session - 1)
                    elif blocks_increase.collidepoint(mouse_pos):
                        # Increase blocks per session (with max limit)
                        self.blocks_per_session = min(MAX_BLOCKS_PER_SESSION, self.blocks_per_session + 1)
                    elif back_button.collidepoint(mouse_pos):
                        # Return to menu
                        self.game_state = "menu"
                
                # Keyboard shortcuts for settings
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # Escape to return to menu
                        self.game_state = "menu"
                    # Toggle settings
                    elif event.key == pygame.K_v:  # V key to toggle visual mode
                        self.visual_only_mode = not self.visual_only_mode
                    elif event.key == pygame.K_r:  # R key to toggle realtime score
                        self.show_realtime_score = not self.show_realtime_score
                    # Value adjustment with arrow keys
                    elif event.key in (pygame.K_UP, pygame.K_RIGHT):
                        # Increase values with Up/Right
                        self.trial_duration = min(MAX_TRIAL_DURATION, self.trial_duration + 0.5)
                    elif event.key in (pygame.K_DOWN, pygame.K_LEFT):
                        # Decrease values with Down/Left
                        self.trial_duration = max(MIN_TRIAL_DURATION, self.trial_duration - 0.5)
                        
            elif self.game_state == "history":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    buttons, total_pages = self.draw_history()
                    
                    # Check if any button was clicked
                    for i, button in enumerate(buttons):
                        if button.collidepoint(mouse_pos):
                            # First button is either "Previous" or "Back" if on first page
                            if i == 0 and self.history_page > 0:
                                self.history_page -= 1
                            # Middle button is always "Back"
                            elif (i == 0 and self.history_page == 0) or (i == 1 and len(buttons) > 1):
                                self.game_state = "menu"
                            # Last button is "Next" if there is more than one page
                            elif (i == 1 and len(buttons) == 2) or i == 2:
                                if self.history_page < total_pages - 1:
                                    self.history_page += 1
                            break
                
                # Keyboard navigation for history
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # Escape to return to menu
                        self.game_state = "menu"
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:  # Left arrow or A for previous page
                        if self.history_page > 0:
                            self.history_page -= 1
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:  # Right arrow or D for next page
                        _, total_pages = self.draw_history()
                        if self.history_page < total_pages - 1:
                            self.history_page += 1
            
            elif self.game_state == "playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == self.position_match_key:  # Position match key (A)
                        self.process_user_input(position_key=True)
                    elif event.key == self.sound_match_key:  # Sound match key (L)
                        self.process_user_input(sound_key=True)
                    elif event.key == pygame.K_SPACE:  # Space can be used for both
                        self.process_user_input(position_key=True, sound_key=True)
                    elif event.key == pygame.K_ESCAPE:  # Escape to return to menu
                        self.game_state = "menu"
                    elif event.key == pygame.K_p:  # P key to pause game
                        self.paused = True
                
                # Also support mouse clicks on the control buttons
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    grid_left, grid_top, grid_width, grid_height = self.draw_grid()
                    
                    # Get control button rects
                    position_button_rect = pygame.Rect(grid_left, grid_top + grid_height + 20, 
                                                    grid_width // 2 - 10, 40)
                    sound_button_rect = pygame.Rect(grid_left + grid_width // 2 + 10, grid_top + grid_height + 20,
                                                 grid_width // 2 - 10, 40)
                    
                    if position_button_rect.collidepoint(mouse_pos):
                        self.process_user_input(position_key=True)
                    elif sound_button_rect.collidepoint(mouse_pos):
                        self.process_user_input(sound_key=True)
            
            elif self.game_state == "block_summary":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    continue_button = self.draw_block_summary()
                    
                    if continue_button.collidepoint(mouse_pos):
                        if self.current_block < self.blocks_per_session:
                            self.start_block()
                            self.game_state = "playing"
                        else:
                            self.game_state = "session_summary"
                
                # Keyboard shortcut for continuing
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if self.current_block < self.blocks_per_session:
                            self.start_block()
                            self.game_state = "playing"
                        else:
                            self.game_state = "session_summary"
            
            elif self.game_state == "session_summary":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    menu_button = self.draw_session_summary()
                    
                    if menu_button.collidepoint(mouse_pos):
                        self.game_state = "menu"
                
                # Keyboard shortcut for returning to menu
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                        self.game_state = "menu"
    
    def update(self):
        # Get time since last frame
        dt = self.clock.get_time() / 1000.0  # Convert to seconds
        
        if self.game_state == "playing" and not self.paused:
            # Update trial timer
            self.trial_timer += dt
            
            # Update feedback timer if it's active
            if self.feedback_timer > 0:
                self.feedback_timer -= dt
                # Reset flags when timer expires
                if self.feedback_timer <= 0:
                    self.missed_position_match = False
                    self.missed_sound_match = False
            
            # Check if trial duration has elapsed
            if self.trial_timer >= self.trial_duration:
                self.end_trial()
    
    def render(self):
        self.screen.fill(BLACK)
        
        if self.game_state == "menu":
            self.draw_menu()
        
        elif self.game_state == "settings":
            self.draw_settings()
            
        elif self.game_state == "history":
            self.draw_history()
        
        elif self.game_state == "playing":
            # Draw game elements
            grid_left, grid_top, grid_width, grid_height = self.draw_grid()
            self.draw_session_info()
            self.draw_controls(grid_left, grid_top, grid_width, grid_height)
            
            # Draw trial timer progress bar
            progress = self.trial_timer / self.trial_duration
            bar_width = self.window_width - 40
            filled_width = int(bar_width * progress)
            
            pygame.draw.rect(self.screen, GRAY, (20, self.window_height - 30, bar_width, 10))
            pygame.draw.rect(self.screen, GREEN, (20, self.window_height - 30, filled_width, 10))
            
            # Add pause button and hint
            pause_hint = self.small_font.render("Press 'P' to pause", True, WHITE)
            self.screen.blit(pause_hint, (self.window_width - pause_hint.get_width() - 20, self.window_height - 50))
            
            # If game is paused, draw pause screen on top
            if self.paused:
                self.draw_pause_screen()
        
        elif self.game_state == "block_summary":
            self.draw_block_summary()
        
        elif self.game_state == "session_summary":
            self.draw_session_summary()
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.render()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = DualNBack()
    game.run()