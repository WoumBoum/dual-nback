# Dual N-Back Memory Training Game

A Python implementation of the Dual N-Back cognitive training task that aims to improve working memory.

![Dual N-Back Game](https://github.com/username/dual-nback/blob/main/screenshots/gameplay.png)

## Overview

Dual N-Back is a challenging cognitive task that requires simultaneously tracking two independent streams of stimuli (visual positions and letters) and identifying when current stimuli match those presented exactly N positions back in the sequence.

## Features

- Visual stream: A square appearing in one of 9 positions (3×3 grid)
- Visual letter display: Letters shown directly in the active square
- Adaptive difficulty: N-level increases or decreases based on performance
- Performance tracking with d-prime sensitivity index
- Real-time score display including d-prime statistics (optional)
- Missed match indicators that show when you fail to respond to a match
- Visual feedback for all correct and incorrect responses
- Pause functionality during active gameplay with 'P' key
- Customizable session length:
  - Adjustable number of trials per block (5-50)
  - Adjustable number of blocks per session (1-30)
- Performance graphs:
  - Accuracy trends for position and letter matching
  - D-prime statistics tracking
  - N-level progression visualization
- Progress history tracking:
  - Detailed session logs saved to 'logs' directory
  - Historical performance graphs showing improvement over time
  - Day-based data grouping for consistent time intervals
  - Moving average trend lines showing long-term progress
  - Multi-page history view with navigation controls
- User data saved between sessions
- Resizable window for different screen sizes
- Settings panel with toggle options
- Adjustable trial duration

## Requirements

- Python 3.6+
- Pygame
- Matplotlib (for performance graphs)
- NumPy (dependency for matplotlib)

## Installation

1. Clone this repository:
```
git clone https://github.com/username/dual-nback.git
cd dual-nback
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
```
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

4. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Game

Make sure you have activated the virtual environment, then run:

```
python dual_nback.py
```

## How to Play

1. In each trial, you'll see a square appear in one of 9 grid positions with a letter displayed inside it
2. Your task is to identify if the current position matches the position from N trials back
3. You also need to identify if the current letter matches the letter from N trials back
4. Press the 'A' key if you detect a position match
5. Press the 'L' key if you detect a letter match
6. Press the SPACE key if both position and letter match
7. Don't press any key if neither matches
8. You can also use mouse clicks on the buttons shown at the bottom of the game screen

## Keyboard Controls

- 'A' key for position matches
- 'L' key for letter matches
- SPACE key for both position and letter matches
- ENTER key to proceed through menus
- ESC key to return to menu
- 'S' key to access settings
- 'H' key to view progress history
- 'V' key to toggle visual-only mode
- 'R' key to toggle real-time score display
- 'P' key to pause/resume during gameplay
- Arrow keys to adjust values in settings
- Left/Right arrows to navigate history pages

## Game Settings

The game includes a comprehensive settings panel with a two-column layout:

### Game Settings (Left Column)
1. **Visual-only Mode** - Toggle between showing letters in the center of active squares (ON) or in a separate display (OFF)
2. **Real-time Score Display** - Toggle showing your current performance metrics during gameplay, including:
   - Position accuracy and d-prime
   - Letter accuracy and d-prime
3. **Trial Duration** - Adjust how long each stimulus is shown (1.5-5 seconds)
4. **Trials per Block** - Set the number of trials in each block (5-50 trials)
5. **Blocks per Session** - Set the number of blocks in a session (1-30 blocks)

### Information (Right Column)
1. **N-Level Adjustment Rules** - Detailed explanation of when N-level changes:
   - Increases when both accuracies ≥ 80% AND both d-primes ≥ 2.0
   - Decreases when either accuracy ≤ 50% OR both d-primes < 1.0
2. **D-Prime Explanation** - Comprehensive guide to understanding d-prime:
   - How hit rates and false alarm rates are calculated
   - What different d-prime values mean for performance
   - Interpretation guide for different d-prime thresholds

## Building an Executable (Optional)

You can build a standalone executable using cx_Freeze:

```
python setup.py build
```

## Contributing

Contributions are welcome! Here are ways you can contribute to this project:

1. Report bugs and issues
2. Suggest new features or improvements
3. Submit pull requests with bug fixes or features
4. Improve documentation

## License

This project is open source and available under the MIT License.