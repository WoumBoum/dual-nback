# Dual N-Back: Core Implementation Specification

## Concept
Dual N-Back is a working memory training task requiring users to simultaneously track two independent streams of stimuli (visual positions and auditory inputs) and identify when current stimuli match those presented exactly N positions back in the sequence.

## Core Mechanics

### Stimuli Handling
- **Visual stream**: Display a square in one of 9 positions (3×3 grid)
- **Auditory stream**: Play one of 8-9 distinct sounds or letters (common set: C, H, K, L, Q, R, S, T)
- **Presentation rate**: Default 3 seconds per trial, configurable between 1.5-5 seconds
- **Session length**: 20+ trials per block, typically 20 blocks per session

### Game Logic
1. For each trial, randomly select and present both visual and auditory stimuli
2. Track stimulus history for both streams in separate arrays
3. Compare current stimuli against those N positions back in their respective arrays
4. Register user input (position match, sound match, both, or none)
5. Calculate accuracy metrics separately for each stream

### User Input
- Require distinct input methods for each stream (keyboard keys, screen buttons)
- Input window should span from stimulus presentation until next stimulus begins
- Register true positives, false positives, true negatives, and false negatives

## Technical Requirements

### Data Structures
- Circular buffers (length N+1) for each stimulus stream
- Trial data structure containing {position_index, sound_index, position_match, sound_match}

### Adaptive Difficulty
- Implement N-level progression based on performance thresholds:
  - Increase N when accuracy ≥ 80% (both streams)
  - Decrease N when accuracy ≤ 50% (either stream)

### Performance Tracking
- Calculate d-prime sensitivity index (signal detection theory)
- Track separate metrics for visual and auditory performance
- Store session data for progress visualization

## User Experience
- Clean, distraction-free interface
- Clear visual and audio feedback for correct/incorrect responses
- Session progress indicator
- Between-block performance summary

## Storage
- Save user progress across sessions
- Track performance metrics over time (N-level achieved, accuracy rates)
