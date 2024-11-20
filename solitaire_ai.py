import pygame
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
import time
import os
import json
import logging
from numba import jit, float32, int32
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
# import the library for pathlib
from pathlib import Path
# Initialize Pygame
pygame.init()

# Game Window Settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CARD_WIDTH = 71
CARD_HEIGHT = 96
FPS = 30

# Colors (RGB Values)
WHITE = (255, 255, 255)    # Used for card faces
GREEN = (34, 139, 34)      # Table background color
BLACK = (0, 0, 0)          # Used for text and card borders
EMPTY_PILE_COLOR = (200, 200, 200)  # Light gray for empty pile outlines

# Game Rules
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
MAX_STOCK_PASSES = 3
# Create lookup dictionaries
value_to_index = {value: idx for idx, value in enumerate(VALUES)}
suit_to_index = {suit: idx for idx, suit in enumerate(SUITS)}


# AI Training Settings
MAX_EPISODES = 1000     # More training episodes
MAX_MOVES_PER_EPISODE = 500  # Maximum moves allowed per game before forcing end
BATCH_SIZE = 4096          # Larger batch size for better learning
MAX_CONSECUTIVE_INVALID_MOVES = 40  # Stop game if this many invalid moves in a row
EXPLORATION_RATE = 0.05     # 5% chance to make random move instead of using model
INITIAL_EPSILON = 1.0     # Start fully exploratory
EPSILON_MIN = 0.05        # Allow more exploration (up from 0.01)
EPSILON_DECAY = 0.997     # Slower decay for better exploration
LEARNING_RATE = 0.001    # Smaller learning rate for more stable learning
GAMMA = 0.95              # Discount rate for future rewards
MEMORY_SIZE = 100000       # Larger memory for better experience replay

# Display Settings
DISPLAY_FREQUENCY = 2      # Show gameplay every N episodes
STATS_FREQUENCY = 1        # Show statistics every N episodes
FRAME_DELAY = 10            # Milliseconds to wait between frames
EPISODE_DELAY = 0.001       # Seconds to wait between episodes

# Neural Network Settings
NN_LAYER_1_SIZE = 1024     # Neurons in first hidden layer
NN_LAYER_2_SIZE = 512     # Neurons in second hidden layer
NN_LAYER_3_SIZE = 256     # New layer
CARDS_PER_TABLEAU = 13    # Maximum cards per tableau pile
TABLEAU_PILES = 7         # Number of tableau piles
VALUES_PER_CARD = 2       # Each card encoded as (value, suit)

# State Size Calculations
TABLEAU_STATE_SIZE = TABLEAU_PILES * CARDS_PER_TABLEAU * VALUES_PER_CARD  # 7 * 13 * 2 = 182
FOUNDATION_STATE_SIZE = 4 * 2      # 4 foundation piles × 2 values per card = 8
WASTE_STATE_SIZE = 2              # 1 waste card × 2 values = 2
STOCK_STATE_SIZE = 1              # 1 value for stock size = 1
PASSES_STATE_SIZE = 1             # 1 value for number of passes = 1

# Total state size for neural network input
STATE_SIZE = (TABLEAU_STATE_SIZE + 
              FOUNDATION_STATE_SIZE + 
              WASTE_STATE_SIZE + 
              STOCK_STATE_SIZE + 
              PASSES_STATE_SIZE)   # Should equal 194

ACTION_SIZE = 50          # Maximum number of possible moves to choose from

# Reward Settings - Encourages AI to:
# 1. Prioritize foundation moves and revealing cards (25 points each)
# 2. Build useful tableau sequences (10 points)
# 3. Draw new cards when stuck (2 points)
# 4. Make basic moves when nothing better is available (1 point)
# 5. Avoid repetitive moves (-5 points)
REWARD_FOUNDATION_MOVE = 15    # Moving card to foundation (Ace piles)
REWARD_REVEAL_CARD = 100        # Revealing a face-down card
REWARD_PRODUCTIVE_TABLEAU = 100  # Building useful sequences in tableau
REWARD_BASIC_MOVE = 1          # Any legal move that doesn't fit above categories
REWARD_DRAW_CARD = -5          # Drawing new card from stock
REWARD_CYCLE_PENALTY = -15      # Penalty for moving same card back and forth
REWARD_LONG_CYCLE_PENALTY = -15 # Penalty for longer sequences of repetitive moves
REWARD_WIN_BONUS = 10000        # Large bonus for winning the game
REWARD_MOVE_FROM_WASTE = 50     # Moving card from waste pile to tableau or foundation

# Save/Load Settings
MODEL_SAVE_PATH = Path("G:/My Drive/solitaire")  # Base path for saving models
SAVE_FREQUENCY = 150           # Save model every N episodes
LOAD_EXISTING_MODEL = True    # Whether to load previous model or start fresh
EPSILON_UPDATE_FREQUENCY = 5   # Update exploration rate every N episodes

# Add to constants section
MAX_UNPRODUCTIVE_MOVES = 50  # Maximum moves allowed without revealing card or foundation move or draw move
MAX_KING_SHUFFLES = 1     # Maximum times to move a king between empty spots
KING_SHUFFLE_PENALTY = -50  # Penalty for moving kings between empty spots
from numba import jit, float32, int32
import numpy as np
@jit(nopython=True)
def calculate_state_fast(tableau_data, foundation_data, waste_data, stock_size):
    """
        Optimized state calculation using Numba
        Parameters:
        tableau_data: 3D array (7, 13, 2) for 7 piles, 13 cards max, 2 values (value, suit)
        foundation_data: 2D array (4, 2) for 4 foundations, 2 values each
        waste_data: 1D array (2,) for top waste card
        stock_size: integer
    """
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    idx = 0
    
    # Encode tableau
    for pile_idx in range(tableau_data.shape[0]):  # 7 piles
        for card_idx in range(tableau_data.shape[1]):  # 13 cards max
            if card_idx < CARDS_PER_TABLEAU:
                value = tableau_data[pile_idx][card_idx][0]
                suit = tableau_data[pile_idx][card_idx][1]
                if value > 0:  # Card exists and is face up
                    state[idx] = value / 13.0
                    state[idx + 1] = suit / 3.0
                idx += 2
    
    # Encode foundations
    for pile_idx in range(foundation_data.shape[0]):
        if foundation_data[pile_idx][0] > 0:  # If pile not empty
            state[idx] = foundation_data[pile_idx][0] / 13.0
            state[idx + 1] = foundation_data[pile_idx][1] / 3.0
        idx += 2
    
    # Encode waste
    if waste_data[0] > 0:  # If waste not empty
        state[idx] = waste_data[0] / 13.0
        state[idx + 1] = waste_data[1] / 3.0
    idx += 2
    
    # Encode stock size
    state[idx] = stock_size / 24.0  # Normalize by max possible stock size
    
    return state
    
class Card:
    images = {}  # Class-level image cache
    
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.face_up = False
        self.image = None
        self.load_image()
    def load_image(self):
        if self.face_up:
            # Create key for face-up card
            key = (self.value, self.suit)
            if key in Card.images:
                self.image = Card.images[key]
            else:
                # Create and cache new face-up card image
                self.image = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
                self.image.fill(WHITE)
                # Draw card border
                pygame.draw.rect(self.image, BLACK, (0, 0, CARD_WIDTH, CARD_HEIGHT), 2)
                
                # Draw card value and suit using basic ASCII
                text = str(self.value)
                if self.suit in ['hearts', 'diamonds']:
                    color = (255, 0, 0)  # Red for hearts and diamonds
                else:
                    color = (0, 0, 0)    # Black for clubs and spades
                suit_symbol = {'hearts': 'H', 'diamonds': 'D', 'clubs': 'C', 'spades': 'S'}[self.suit]
                font = pygame.font.Font(None, 36)
                
                # Draw value
                value_text = font.render(text, True, color)
                self.image.blit(value_text, (5, 5))
                
                # Draw suit
                suit_text = font.render(suit_symbol, True, color)
                self.image.blit(suit_text, (5, CARD_HEIGHT - 30))
                
                # Add center suit for better visibility
                center_text = font.render(suit_symbol, True, color)
                center_x = (CARD_WIDTH - center_text.get_width()) // 2
                center_y = (CARD_HEIGHT - center_text.get_height()) // 2
                self.image.blit(center_text, (center_x, center_y))
                
                # Cache the created image
                Card.images[key] = self.image
        else:
            # Handle card back image
            if 'back' in Card.images:
                self.image = Card.images['back']
            else:
                # Create and cache new card back image
                self.image = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
                self.image.fill((50, 50, 200))  # Blue back
                # Add card back pattern
                pygame.draw.rect(self.image, BLACK, (0, 0, CARD_WIDTH, CARD_HEIGHT), 2)
                for i in range(0, CARD_HEIGHT, 10):
                    pygame.draw.line(self.image, (40, 40, 180), (5, i), (CARD_WIDTH-5, i), 1)
                # Cache the card back image
                Card.images['back'] = self.image

    @classmethod
    def clear_cache(cls):
        """Clear the image cache if needed (e.g., when changing card styles)"""
        cls.images.clear()

class SolitaireGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("AI Solitaire")
        self.clock = pygame.time.Clock()
        self.reset_game()
        self.move_history = []
        self.last_from_pile = None
        self.last_to_pile = None
        self.stock_passes = 0
        self.king_shuffles = 0
    def reset_game(self):
        # Initialize deck
        self.deck = [Card(suit, value) for suit in SUITS for value in VALUES]
        random.shuffle(self.deck)
        
        # Initialize game state
        self.tableau = [[] for _ in range(7)]
        self.foundations = [[] for _ in range(4)]
        self.waste = []
        self.stock = []
        
        # Deal initial cards to tableau
        for i in range(7):  # For each tableau pile
            for j in range(i, 7):  # Deal cards from i to 6
                card = self.deck.pop()
                if j == i:  # If it's the bottom card of the pile
                    card.face_up = True
                    card.load_image()  # Make sure to load the face-up image
                else:
                    card.face_up = False
                    card.load_image()
                self.tableau[j].append(card)
                
        # Remaining cards go to stock, all face down
        for card in self.deck:
            card.face_up = False
            card.load_image()
        self.stock = self.deck
        
        # Reset move tracking
        self.move_history = []
        self.last_from_pile = None
        self.last_to_pile = None
        self.stock_passes = 0  # Reset stock passes counter
        self.king_shuffles = 0
                



    def get_state(self):
        # Convert game state to numpy arrays for Numba
        tableau_data = np.zeros((7, CARDS_PER_TABLEAU, 2), dtype=np.int32)
        foundation_data = np.zeros((4, 2), dtype=np.int32)
        waste_data = np.zeros(2, dtype=np.int32)
        
        # Fill tableau data
        for i, pile in enumerate(self.tableau):
            for j, card in enumerate(pile[:CARDS_PER_TABLEAU]):
                if card.face_up:
                    tableau_data[i][j][0] = value_to_index[card.value] + 1
                    tableau_data[i][j][1] = suit_to_index[card.suit]
        
        # Fill foundation data
        for i, pile in enumerate(self.foundations):
            if pile:
                foundation_data[i][0] = value_to_index[pile[-1].value] + 1
                foundation_data[i][1] = suit_to_index[pile[-1].suit]
        
        # Fill waste data
        if self.waste:
            waste_data[0] = value_to_index[self.waste[-1].value] + 1
            waste_data[1] = suit_to_index[self.waste[-1].suit]
        
        return calculate_state_fast(tableau_data, foundation_data, waste_data, len(self.stock))

    def get_valid_moves(self):
        moves = []
        revealed_moves = []
        productive_moves = []
        
        # First priority: Moves that reveal face-down cards
        for i, tableau_pile in enumerate(self.tableau):
            if tableau_pile and tableau_pile[-1].face_up:
                if len(tableau_pile) > 1 and not tableau_pile[-2].face_up:
                    for j, to_pile in enumerate(self.tableau):
                        if i != j and self._can_move_to_tableau(tableau_pile[-1], to_pile):
                            revealed_moves.append(('tableau_to_tableau', i, j, len(tableau_pile)-1))
                    for j, foundation in enumerate(self.foundations):
                        if self._can_move_to_foundation(tableau_pile[-1], foundation):
                            revealed_moves.append(('tableau_to_foundation', i, j))

        # Second priority: Moves to foundation
        if self.waste:
            card = self.waste[-1]
            for j, foundation in enumerate(self.foundations):
                if self._can_move_to_foundation(card, foundation):
                    productive_moves.append(('waste_to_foundation', None, j))

        for i, tableau_pile in enumerate(self.tableau):
            if tableau_pile and tableau_pile[-1].face_up:
                card = tableau_pile[-1]
                for j, foundation in enumerate(self.foundations):
                    if self._can_move_to_foundation(card, foundation):
                        productive_moves.append(('tableau_to_foundation', i, j))

        # Third priority: Productive tableau moves (moving cards to build on other cards)
        if self.waste:
            card = self.waste[-1]
            for j, tableau in enumerate(self.tableau):
                if self._can_move_to_tableau(card, tableau) and (not tableau or 
                    VALUES.index(card.value) == VALUES.index(tableau[-1].value) - 1):
                    productive_moves.append(('waste_to_tableau', None, j))

        for i, from_pile in enumerate(self.tableau):
            if from_pile and from_pile[-1].face_up:
                moveable_indices = self._get_moveable_cards(from_pile)
                for start_idx in moveable_indices:
                    moving_card = from_pile[start_idx]
                    for j, to_pile in enumerate(self.tableau):
                        if i != j and self._can_move_to_tableau(moving_card, to_pile):
                            # Only add if it's not moving to an empty space or if it's a King
                            if (to_pile and VALUES.index(moving_card.value) == VALUES.index(to_pile[-1].value) - 1) or \
                               (not to_pile and moving_card.value == 'K'):
                                productive_moves.append(('tableau_to_tableau', i, j, start_idx))

        # Last priority: Draw move
        # Always add draw move if:
        # 1. There are cards in stock, OR
        # 2. There are cards in waste and we haven't exceeded passes
        if self.stock:  # Can draw from stock
            moves.append(('draw', None, None))
        elif self.waste and self.stock_passes < MAX_STOCK_PASSES:  # Can recycle waste
            moves.append(('draw', None, None))
            
        return revealed_moves + productive_moves + moves

    def _can_move_to_foundation(self, card, foundation):
        if not foundation:  # If foundation is empty
            return card.value == 'A'  # Only aces can start a foundation
        top_card = foundation[-1]
        # Check if same suit and next value
        return (card.suit == top_card.suit and 
                VALUES.index(card.value) == VALUES.index(top_card.value) + 1)
    
    def _can_move_to_tableau(self, card, tableau_pile):
        if not card.face_up:  # Never allow moving face-down cards
            return False
        if not tableau_pile:  # If tableau pile is empty
            return card.value == 'K'  # Only kings can start a new pile
        top_card = tableau_pile[-1]
        if not top_card.face_up:
            return False
        # Check if alternate color and one value less
        is_red = card.suit in ['hearts', 'diamonds']
        top_is_red = top_card.suit in ['hearts', 'diamonds']
        return (is_red != top_is_red and 
                VALUES.index(card.value) == VALUES.index(top_card.value) - 1)

    def _reveal_top_card(self, tableau_pile):
        """Helper method to reveal the top card of a tableau pile"""
        if tableau_pile and not tableau_pile[-1].face_up:
            # print(f"Revealing card in pile") # Debug print
            tableau_pile[-1].face_up = True
            tableau_pile[-1].load_image()
            return True
        return False

    def _get_moveable_cards(self, tableau_pile):
        """Returns a list of indices of face-up cards that can be moved as a stack"""
        if not tableau_pile:
            return []
        
        moveable = []
        # Start from the end and work backwards until we hit a face-down card
        for i in range(len(tableau_pile)-1, -1, -1):
            if tableau_pile[i].face_up:
                moveable.insert(0, i)
            else:
                break
        return moveable

    def make_move(self, move):
        if not move:
            return -1

        move_type, from_idx, to_idx, *extra_args = (*move, None)
        
        # Check for unproductive cycling
        if (move_type == 'tableau_to_tableau' and 
            from_idx == self.last_to_pile and 
            to_idx == self.last_from_pile):
            return REWARD_CYCLE_PENALTY
            
        # Store move for cycle detection
        self.move_history.append(move)
        if len(self.move_history) > 4:
            self.move_history.pop(0)
            
        # Check for longer cycles
        if len(self.move_history) >= 4:
            if self._is_cycle(self.move_history[-4:]):
                return REWARD_LONG_CYCLE_PENALTY
        
        reward = 0
        
        if move_type == 'waste_to_foundation':
            if not self.waste:
                return -1
            card = self.waste.pop()
            self.foundations[to_idx].append(card)
            reward = REWARD_FOUNDATION_MOVE + REWARD_MOVE_FROM_WASTE
            
        elif move_type == 'waste_to_tableau':
            if not self.waste:
                return -1
            card = self.waste.pop()
            self.tableau[to_idx].append(card)
            reward = REWARD_PRODUCTIVE_TABLEAU + REWARD_MOVE_FROM_WASTE

        elif move_type == 'tableau_to_foundation':
            if not self.tableau[from_idx]:
                return -1
            card = self.tableau[from_idx][-1]  # Get the card without popping first
            if not card.face_up:
                return -1
            
            # Verify move is valid before making it
            if not self._can_move_to_foundation(card, self.foundations[to_idx]):
                return -1
                
            # Now make the move
            self.tableau[from_idx].pop()
            self.foundations[to_idx].append(card)
            reward = REWARD_FOUNDATION_MOVE
            
            # Always check for card to reveal after removing a card
            if self._reveal_top_card(self.tableau[from_idx]):
                reward += REWARD_REVEAL_CARD
            
        elif move_type == 'tableau_to_tableau':
            if not self.tableau[from_idx]:
                return -1
            
            start_idx = extra_args[0] if extra_args else len(self.tableau[from_idx]) - 1
            moving_cards = self.tableau[from_idx][start_idx:]
            
            # Check for king shuffling
            if (moving_cards[0].value == 'K' and 
                not self.tableau[to_idx] and 
                len(moving_cards) == 1):  # Only single king
                self.king_shuffles += 1
                if self.king_shuffles > MAX_KING_SHUFFLES:
                    return KING_SHUFFLE_PENALTY
            else:
                self.king_shuffles = 0  # Reset counter for non-king moves
            
            # Check if move would exceed pile size limit
            if len(self.tableau[to_idx]) + len(moving_cards) > CARDS_PER_TABLEAU:
                return -1  # Invalid move - would exceed size limit
            
            # Strict validation for face-down cards
            if any(not card.face_up for card in moving_cards):
                print(f"Warning: Attempted to move face-down card from pile {from_idx}")
                return -1
                
            # Verify the move is valid
            if not self._can_move_to_tableau(moving_cards[0], self.tableau[to_idx]):
                return -1
                
            # Before moving, check if we'll reveal a card
            will_reveal = (len(self.tableau[from_idx]) > start_idx + 1 and 
                         not self.tableau[from_idx][start_idx-1].face_up)
                
            # Make the move
            self.tableau[from_idx] = self.tableau[from_idx][:start_idx]
            self.tableau[to_idx].extend(moving_cards)
            
            # Reveal next card if available
            if self._reveal_top_card(self.tableau[from_idx]):
                self.tableau[from_idx][-1].load_image()  # Ensure the newly revealed card is displayed
                reward += REWARD_REVEAL_CARD
            elif will_reveal:
                reward += REWARD_PRODUCTIVE_TABLEAU
            else:
                reward += REWARD_BASIC_MOVE
                
            # Update last move tracking
            self.last_from_pile = from_idx
            self.last_to_pile = to_idx
            
        elif move_type == 'draw':
            if self.stock:  # Draw from stock
                card = self.stock.pop()
                card.face_up = True
                card.load_image()
                self.waste.append(card)
                return REWARD_DRAW_CARD
            elif self.waste and self.stock_passes < MAX_STOCK_PASSES:  # Recycle waste
                # print(f"Recycling waste pile (Pass {self.stock_passes + 1}/{MAX_STOCK_PASSES})")
                while self.waste:
                    card = self.waste.pop()
                    card.face_up = False
                    card.load_image()
                    self.stock.append(card)
                self.stock_passes += 1
                return 0
            else:  # Invalid move
                if self.stock_passes >= MAX_STOCK_PASSES:
                    print(f"Cannot draw: Maximum passes reached ({self.stock_passes}/{MAX_STOCK_PASSES})")
                return -1
                
        return reward

    def _is_cycle(self, moves):
        """Check if a sequence of moves forms a cycle"""
        if len(moves) < 2:
            return False
        # Check if we're moving the same cards back and forth
        for i in range(len(moves)-1):
            if (moves[i][0] == moves[i+1][0] == 'tableau_to_tableau' and
                moves[i][1] == moves[i+1][2] and
                moves[i][2] == moves[i+1][1]):
                return True
        return False

    def draw(self):
        self.screen.fill(GREEN)
        
        # Draw tableau
        for i, pile in enumerate(self.tableau):
            # Draw empty tableau pile outline
            x = 50 + i * (CARD_WIDTH + 20)
            y = 150
            pygame.draw.rect(self.screen, EMPTY_PILE_COLOR, (x, y, CARD_WIDTH, CARD_HEIGHT), 2)
            
            for j, card in enumerate(pile):
                y = 150 + j * 20
                self.screen.blit(card.image, (x, y))
                
        # Draw foundations
        for i, pile in enumerate(self.foundations):
            x = 50 + i * (CARD_WIDTH + 20)
            y = 20
            # Draw empty foundation outline
            pygame.draw.rect(self.screen, EMPTY_PILE_COLOR, (x, y, CARD_WIDTH, CARD_HEIGHT), 2)
            if pile:  # Draw the top card if pile has cards
                self.screen.blit(pile[-1].image, (x, y))
                
        # Draw stock and waste
        stock_x = WINDOW_WIDTH - (CARD_WIDTH + 70)
        waste_x = WINDOW_WIDTH - (2 * CARD_WIDTH + 90)
        
        # Draw stock outline and cards
        pygame.draw.rect(self.screen, EMPTY_PILE_COLOR, (stock_x, 20, CARD_WIDTH, CARD_HEIGHT), 2)
        if self.stock:
            self.screen.blit(self.stock[-1].image, (stock_x, 20))
            # Draw stock count
            font = pygame.font.Font(None, 24)
            count_text = font.render(f"Stock: {len(self.stock)}", True, BLACK)
            self.screen.blit(count_text, (stock_x + 5, 5))
            
            # Draw passes count separately
            passes_text = font.render(f"Passes: {self.stock_passes}/{MAX_STOCK_PASSES}", True, BLACK)
            self.screen.blit(passes_text, (stock_x + 5, CARD_HEIGHT + 25))
        else:
            # Show passes when stock is empty
            font = pygame.font.Font(None, 24)
            passes_text = font.render(f"Passes: {self.stock_passes}/{MAX_STOCK_PASSES}", True, BLACK)
            self.screen.blit(passes_text, (stock_x + 5, CARD_HEIGHT + 25))
        
        # Draw waste outline and cards
        pygame.draw.rect(self.screen, EMPTY_PILE_COLOR, (waste_x, 20, CARD_WIDTH, CARD_HEIGHT), 2)
        if self.waste:
            self.screen.blit(self.waste[-1].image, (waste_x, 20))
            # Show waste count
            font = pygame.font.Font(None, 24)
            waste_text = font.render(f"Waste: {len(self.waste)}", True, BLACK)
            self.screen.blit(waste_text, (waste_x + 5, 5))
        
        # Draw moves counter
        font = pygame.font.Font(None, 24)
        moves_text = font.render(f"Moves: {self.moves_made}", True, BLACK)
        moves_rect = moves_text.get_rect(bottomleft=(10, WINDOW_HEIGHT - 10))
        self.screen.blit(moves_text, moves_rect)
        
        # If game is over, display game over message
        game_over, reason = self.is_game_over()
        if game_over:
            font = pygame.font.Font(None, 48)
            if reason == "win":
                text = font.render("Game Won!", True, (0, 255, 0))
            elif reason == "stuck":
                text = font.render("No More Moves", True, (255, 0, 0))
            elif reason == "move_limit":
                text = font.render("Move Limit Reached", True, (255, 165, 0))
            
            text_rect = text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT - 50))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def is_game_over(self):
        """Check if the game is over"""
        # Win condition: all foundations complete
        if all(len(f) == 13 for f in self.foundations):
            return True, "win"
            
        # Check if we've exceeded move limit
        if hasattr(self, 'moves_made') and self.moves_made >= MAX_MOVES_PER_EPISODE:
            return True, "move_limit"
            
        # Check if we can still make moves
        if self.stock:  # Can still draw from stock
            return False, None
            
        if self.waste and self.stock_passes < MAX_STOCK_PASSES:  # Can still recycle waste
            return False, None
            
        # Check for any other valid moves (excluding draw moves)
        moves = self.get_valid_moves()
        if moves and not all(move[0] == 'draw' for move in moves):
            return False, None
            
        # If we get here, no valid moves and can't draw
        return True, "stuck"

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = INITIAL_EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.epsilon_updates = 0  # Track number of updates
        
        # Try to load existing model if specified
        if LOAD_EXISTING_MODEL:
            self.model = self.load_model()
        
        # Create new model if loading failed or not requested
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(NN_LAYER_1_SIZE, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(NN_LAYER_2_SIZE, activation='relu'),
            keras.layers.Dense(NN_LAYER_3_SIZE, activation='relu'),
            keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        try:
            if self.action_size == 0:
                return 0
            if random.random() <= self.epsilon:
                return random.randrange(self.action_size)
            # Ensure state has correct shape
            if len(state) != 194:
                print(f"Warning: Incorrect state size in act(). Expected 194, got {len(state)}")
                state = np.pad(state, (0, 194 - len(state))) if len(state) < 194 else state[:194]
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return np.argmax(act_values[0])
        except Exception as e:
            print(f"Error in act(): {e}")
            return random.randrange(self.action_size)

    def replay(self, batch_size):
        try:
            if len(self.memory) < batch_size:
                return
            
            minibatch = random.sample(self.memory, batch_size)
            states = np.array([i[0] for i in minibatch])
            actions = np.array([i[1] for i in minibatch])
            rewards = np.array([i[2] for i in minibatch])
            next_states = np.array([i[3] for i in minibatch])
            dones = np.array([i[4] for i in minibatch])

            # Ensure all states have correct shape
            for i in range(len(states)):
                if len(states[i]) != 194:
                    states[i] = np.pad(states[i], (0, 194 - len(states[i]))) if len(states[i]) < 194 else states[i][:194]
                if len(next_states[i]) != 194:
                    next_states[i] = np.pad(next_states[i], (0, 194 - len(next_states[i]))) if len(next_states[i]) < 194 else next_states[i][:194]

            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # Get Q-values for next states
            target_q_values = self.model.predict(next_states, verbose=0)
            # Calculate targets using Bellman equation
            targets = rewards + self.gamma * np.amax(target_q_values, axis=1) * (1 - dones)
            # Get current Q-values
            targets_full = self.model.predict(states, verbose=0)
            # Update Q-values for taken actions
            ind = np.array([i for i in range(batch_size)])
            targets_full[[ind], [actions]] = targets

            # Train model using single batch
            self.model.train_on_batch(states, targets_full)
            
        except Exception as e:
            print(f"Error in replay(): {e}")
    
    def update_epsilon(self):
        """Update epsilon value with decay"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon_updates += 1

    def save_model(self, episode=None):
        """Save the model and its weights"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            
            # Add episode number to filename if provided
            save_path = f"{MODEL_SAVE_PATH}_{episode}" if episode is not None else MODEL_SAVE_PATH
            self.model.save(f"{save_path}.keras")
            print(f"Model saved to {save_path}.keras")
            
            # Save epsilon value
            with open(f"{save_path}_params.json", 'w') as f:
                json.dump({
                    'epsilon': self.epsilon,
                    'episode': episode
                }, f)
            
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load the model and its weights"""
        try:
            # Find the latest saved model if it exists
            import os
            import glob
            import re
            
            # Look for model files
            model_files = glob.glob(f"{MODEL_SAVE_PATH}*.keras")
            if not model_files:
                print("No saved model found, creating new model")
                return None
                
            # Get the most recent model file
            latest_model = max(model_files, key=os.path.getctime)
            print(f"Loading model from {latest_model}")
            
            # Load the model
            model = keras.models.load_model(latest_model)
            
            # Try to load epsilon and other parameters
            params_file = latest_model.replace('.keras', '_params.json')
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    self.epsilon = params.get('epsilon', self.epsilon)
                    print(f"Restored epsilon: {self.epsilon}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

def main():
    # have the user choose between continuing training or starting a new training session
    user_choice = Prompt.ask("[bold green]Do you want to continue training or start a new training session?[/bold green]")
    # ensure the user choice is valid and accept a variety of inputs
    if user_choice.lower() in ["continue", "c"]:
        LOAD_EXISTING_MODEL = True
    else:
        LOAD_EXISTING_MODEL = False
        
    game = SolitaireGame()
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    
    # If loading existing model, get the starting episode
    start_episode = 0

    if LOAD_EXISTING_MODEL:
        try:
            import glob
            import re
            model_files = glob.glob(f"{MODEL_SAVE_PATH}*_params.json")
            if model_files:
                latest_params = max(model_files, key=os.path.getctime)
                with open(latest_params, 'r') as f:
                    params = json.load(f)
                    start_episode = params.get('episode', 0) + 1
                print(f"Continuing training from episode {start_episode}")
        except Exception as e:
            print(f"Error loading episode number: {e}")
    
    # Add tracking variables
    total_cumulative_reward = 0
    num_episodes_completed = 0
    num_wins = 0  # Add win counter
    
    console = Console()
    
    try:
        for e in range(start_episode, start_episode + MAX_EPISODES):
            game.reset_game()
            state = game.get_state()
            total_reward = 0
            moves_made = 0
            consecutive_invalid_moves = 0
            unproductive_moves = 0
            last_foundation_cards = sum(len(f) for f in game.foundations)
            last_revealed_count = sum(1 for pile in game.tableau for card in pile if card.face_up)
            game.moves_made = 0
            
            # Episode loop
            while moves_made < MAX_MOVES_PER_EPISODE:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.VIDEORESIZE:
                        game.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
                valid_moves = game.get_valid_moves()
                game_over, reason = game.is_game_over()
                
                if game_over:
                    if reason == "win":
                        print(f"Game won in {moves_made} moves!")
                        total_reward += REWARD_WIN_BONUS
                        num_wins += 1  # Increment win counter
                    elif reason == "stuck":
                        print(f"Game stuck with no valid moves after {moves_made} moves")
                    elif reason == "move_limit":
                        print(f"Move limit reached ({MAX_MOVES_PER_EPISODE} moves)")
                    
                    if e % DISPLAY_FREQUENCY == 0:
                        game.draw()
                        pygame.time.wait(1000)
                    break
                
                if not valid_moves or consecutive_invalid_moves > MAX_CONSECUTIVE_INVALID_MOVES:
                    print(f"No valid moves available after {moves_made} moves")
                    break
                
                # Choose and make move
                if random.random() < EXPLORATION_RATE:
                    action = random.randrange(len(valid_moves))
                else:
                    action = agent.act(state) % len(valid_moves)
                    
                move = valid_moves[action]
                move_type = move[0]
                reward = game.make_move(move)
                
                if reward < 0:
                    consecutive_invalid_moves += 1
                else:
                    consecutive_invalid_moves = 0
                
                total_reward += reward
                moves_made += 1
                
                next_state = game.get_state()
                done = all(len(f) == 13 for f in game.foundations)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # Update productivity tracking
                current_foundation_cards = sum(len(f) for f in game.foundations)
                current_revealed_count = sum(1 for pile in game.tableau for card in pile if card.face_up)
                
                was_productive = (
                    current_foundation_cards > last_foundation_cards or
                    current_revealed_count > last_revealed_count or
                    move_type == 'draw'
                )
                
                if was_productive:
                    unproductive_moves = 0
                    last_foundation_cards = current_foundation_cards
                    last_revealed_count = current_revealed_count
                else:
                    unproductive_moves += 1
                    if unproductive_moves >= MAX_UNPRODUCTIVE_MOVES:
                        break
                
                game.moves_made = moves_made
                
                # Display game if needed
                if e % DISPLAY_FREQUENCY == 0:
                    game.draw()
                    pygame.time.wait(FRAME_DELAY)
                    pygame.event.pump()
            
            # End of episode training
            if len(agent.memory) > BATCH_SIZE:
                # Perform multiple training iterations at end of episode
                num_training_iterations = min(moves_made // 10, 10)  # Train more for longer episodes
                for _ in range(num_training_iterations):
                    agent.replay(BATCH_SIZE)
            
            # Update epsilon at end of episode
            if e % EPSILON_UPDATE_FREQUENCY == 0:
                agent.update_epsilon()
            
            # Update statistics
            total_cumulative_reward += total_reward
            num_episodes_completed += 1
            avg_reward = total_cumulative_reward / num_episodes_completed
            
            # Update statistics display using rich
            if e % STATS_FREQUENCY == 0:
                # Create a table for the episode stats
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", justify="right", style="green")
                
                # Add rows with formatted values
                table.add_row("Episode", f"{e}/{start_episode + MAX_EPISODES}")
                table.add_row("Score", f"{total_reward:,}")
                table.add_row("Moves", f"{moves_made}")
                table.add_row("Epsilon", f"{agent.epsilon:.4f}")
                table.add_row("Updates", f"{agent.epsilon_updates}")
                table.add_row("Avg Reward", f"{avg_reward:.2f}")
                
                # Calculate correct win rate
                win_rate = (num_wins / num_episodes_completed) if num_episodes_completed > 0 else 0
                table.add_row("Wins", f"{num_wins}")
                table.add_row("Win Rate", f"{win_rate:.2%}")
                
                # Create a panel to contain the table
                panel = Panel(
                    table,
                    title="[bold blue]Training Statistics[/bold blue]",
                    border_style="blue",
                    padding=(1, 2)
                )
                
                # Clear previous output and display new stats
                console.clear()
                console.print(panel)
                
                # Add a separator for better readability
                console.print("─" * console.width, style="dim")
            
            # Save model periodically
            if e % SAVE_FREQUENCY == 0:
                agent.save_model(episode=e)
                
            time.sleep(EPISODE_DELAY)
            pygame.event.pump()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model(episode=e)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        pygame.quit() 
if __name__ == "__main__":
    main()