import cv2
import copy
import time
import numpy as np
from ModulKlasifikasiCitraCNN2 import LoadModel  

# Load model and constants
model = LoadModel("CardModelWeight.h5")
card_names = [
  "Closed Card","Two Club","Three Club","Four Club","Five Club","Six Club","Seven Club","Eight Club","Nine Club","Ten Club","Jack Club","Queen Club","King Club","Ace Club",
  "Two Heart","Three Heart","Four Heart","Five Heart","Six Heart","Seven Heart","Eight Heart","Nine Heart","Ten Heart","Jack Heart","Queen Heart","King Heart","Ace Heart",
  "Two Spade","Three Spade","Four Spade","Five Spade","Six Spade","Seven Spade","Eight Spade","Nine Spade","Ten Spade","Jack Spade","Queen Spade","King Spade","Ace Spade",
  "Two Diamonds","Three Diamonds","Four Diamonds","Five Diamonds","Six Diamonds","Seven Diamonds","Eight Diamonds","Nine Diamonds","Ten Diamonds","Jack Diamonds","Queen Diamonds","King Diamonds","Ace Diamonds",
  ]

card_values = {
        "Two Club": 2, "Three Club": 3, "Four Club": 4, "Five Club": 5,
        "Six Club": 6, "Seven Club": 7, "Eight Club": 8, "Nine Club": 9,
        "Ten Club": 10, "Jack Club": 10, "Queen Club": 10, "King Club": 10,
        "Ace Club": 11, "Two Heart": 2, "Three Heart": 3, "Four Heart": 4,
        "Five Heart": 5, "Six Heart": 6, "Seven Heart": 7, "Eight Heart": 8,
        "Nine Heart": 9, "Ten Heart": 10, "Jack Heart": 10, "Queen Heart": 10,
        "King Heart": 10, "Ace Heart": 11, "Two Spade": 2, "Three Spade": 3,
        "Four Spade": 4, "Five Spade": 5, "Six Spade": 6, "Seven Spade": 7,
        "Eight Spade": 8, "Nine Spade": 9, "Ten Spade": 10, "Jack Spade": 10,
        "Queen of Spade": 10, "King Spade": 10, "Ace Spade": 11, "Two Diamonds": 2,
        "Three Diamonds": 3, "Four Diamonds": 4, "Five Diamonds": 5, "Six Diamonds": 6,
        "Seven Diamonds": 7, "Eight Diamonds": 8, "Nine Diamonds": 9, "Ten Diamonds": 10,
        "Jack Diamonds": 10, "Queen Diamonds": 10, "King Diamonds": 10,
        "Ace Diamonds": 11, "Closed Card": 0
    }

# Global variables
is_clicking = False
clicked_button = None

# Button areas
hit_button_area = (100, 500, 100, 50)
stand_button_area = (300, 500, 100, 50)
play_again_button_area = (200, 300, 150, 50)

# Game & Card functions
def preprocess_image(frame):
    # ... (same as your original code)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((3,3))
    dial = cv2.dilate(canny, kernel=kernel, iterations=1)
    return dial

def find_contours(frame, original, draw=False):
    # ... (same as your original code)
    contours,_ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    proper = sorted(contours, key=cv2.contourArea, reverse=True)
    corners_list=[]

    for cnt in proper:
         area = cv2.contourArea(cnt)
         perimeter = cv2.arcLength(cnt, closed=True)
     
         if area > 5000:
             approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, closed=True)
         
             if len(approx) == 4:
                 corners = np.float32(approx.reshape(4,2))
                 corners_list.append(corners)
                 if draw:
                     cv2.drawContours(original, [approx], -1, (0,255,0), 2)
                     for corner in corners:
                         x,y = corner[0],corner[1]
                         cv2.circle(original,(int(x),int(y)), 5, (255,0,0), -1)
                     
    return original, corners_list

def cardWrap(frame,corners):
    # ... (same as your original code)
    width,height = 200,300
    if len(corners) == 4:
        pts1 = np.float32(corners)
        pts2 = np.float32([[0,0], [0,height], [width,height], [width,0]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(frame, matrix, (width,height))
        imgOutput = cv2.resize(imgOutput, (128,128))
        
        return imgOutput
    else:
        return frame 

def detect_card(frame):
    # ... (same as your original code)
    FrameResult = frame.copy()
    FrameResult2 = frame.copy()

    preprocessed_frame = preprocess_image(frame)
    processed_frame, corners_list = find_contours(preprocessed_frame, FrameResult, draw=True)
    
    detected_cards = []

    for corners in corners_list:
    # Process each set of corners
        if len(corners) == 4:  # Assuming 4 corners indicate a single card
            wrap_corners = cardWrap(FrameResult2, corners)
        
        # Resize and reshape the wrapped card for prediction
            wrap_corners_resized = cv2.resize(wrap_corners, (128, 128))
            wrap_corners_resized_reshaped = np.expand_dims(wrap_corners_resized, axis=0)
        
        # predictions for each card
            predictions = model.predict(wrap_corners_resized_reshaped)
            predicted_labels = np.argmax(predictions, axis=1)
            predicted_card_names = [card_names[label] for label in predicted_labels]
        
        detected_cards.append((predicted_card_names))
        
        print("Detected cards:", predicted_card_names) #check
        for i, card_name in enumerate(predicted_card_names):
            text_position = (int(corners[0][0]), int(corners[0][1]) - 10)  # Adjust the text position
            cv2.putText(FrameResult, card_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Draw a rectangle
            #cv2.polylines(FrameResult, [np.int32(corners)], True, (0, 255, 0), 2)
            
    if not detected_cards:  # If no cards are detected
        return FrameResult, None # Return None if no cards are detected
    else:
        return FrameResult, detected_cards #return proccedframe

def draw_frame(frame, player_hand, dealer_hand, player_score, dealer_score,):
  """Draws the game frame with split areas for dealer and player, including score displays."""
  # Define frame dimensions
  height, width, _ = frame.shape

   # Calculate split area dimensions
  split_width = width // 2  
  border_thickness = 5

   # Draw vertical border line
  cv2.line(frame, (split_width, 0), (split_width, height), (0, 0, 0), thickness=border_thickness)

  # Draw player area
  player_area = frame[:, :split_width]
  # Calculate updated player score
  player_score = calculate_hand_value(player_hand)
  cv2.putText(frame, f"Player Score: {player_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
  # Draw dealer area
  dealer_area = frame[:, split_width:]
  dealer_score = calculate_hand_value(dealer_hand)
  cv2.putText(frame, f"Dealer Scroe: {dealer_score}", (350,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  

##Game Logic
# Process detected cards and build player and dealer hands

def parse_card_name(card_name):
    """
    Parses a predicted card name into rank and suit using different cases.

    Args:
      card_name: The predicted card name as a string.

    Returns:
      A tuple containing the extracted rank and suit, or None if parsing fails.
    """
    if isinstance(card_name, str):
        parts = card_name.split()
        
        if len(parts) == 2:  # Assuming format like "Two Diamonds"
            rank, suit = parts[0], parts[1]
            return rank, suit
        
        elif len(parts) == 3:  # Handles cases like "Queen of Spade"
            rank = parts[0]
            suit = parts[2] if parts[2] != 'of' else parts[1]
            return rank, suit
        
        elif card_name in ["Ace", "Jack", "Queen", "King"]:  # Handles single-word ranks
            rank = card_name
            suit = None  # Handle missing suit information
            return rank, suit

        else:
            print(f"Warning: Unrecognized card name format: '{card_name}'")
            return None, None
    else:
        print(f"Warning: Invalid input type for card name: {type(card_name)}")
        return None, None

def calculate_hand_value(hand):
    total_score = 0
    has_ace = False

    for card in hand:
        # Ensure the card value is properly formatted
        if isinstance(card, tuple) and len(card) == 2:
            rank, value = card[0], card[1]
            if value == 11:  # Check if the card is an Ace
                has_ace = True
            total_score += value

    # Jika nilai kartu lebih dari 21 dan ada Ace
    # Kurangi 10 dari total skor untuk setiap Ace sampai nilai tidak melebihi 21
    while total_score > 21 and has_ace:
        total_score -= 10
        has_ace = False  # Hanya kurangi satu Ace

    return total_score

    
def handle_detected_cards(detected_cards):
    player_hand = []
    dealer_hand = []
    player_score = 0
    dealer_score = 0

    if detected_cards is not None:  # Check if detections are available
        for card_names in detected_cards:
            for card_name in card_names:
                rank, suit = parse_card_name(card_name)

                if rank != "Closed Card":
                    card_value = card_values.get(f"{rank} {suit}", None)  # Consider both rank and suit for lookup
                    if card_value is None:
                        print(f"Warning: Rank '{rank} {suit}' not found in card_values dictionary")
                    else:
                        card = (rank, card_value)
                        player_hand.append(card)
                        dealer_hand.append(card)
                        print(f"Card: {card_name}, Value: {card_value}")  # Checking cardvvalues

    player_score = calculate_hand_value(player_hand)
    dealer_score = calculate_hand_value(dealer_hand)
    print(f"Player Hand: {player_hand}")  # Check the cards in player's hand
    print(f"Dealer Hand: {dealer_hand}")  # Check the cards in dealer's hand
    print(f"Player Score: {player_score}")  # Check the calculated player score
    print(f"Dealer Score: {dealer_score}")  # Check the calculated dealer score

    return player_hand, dealer_hand, player_score, dealer_score

    return player_hand, dealer_hand, player_score, dealer_score

def check_winner(player_score,dealer_score):
    global winner
    winner = None
    if game_over:
        if player_score > 21:
            winner = "Dealer"
        elif dealer_score > 21:
            winner = "Player"
        elif player_score == 21 and dealer_score != 21:
            winner = "Player" 
        elif player_score > dealer_score:
            winner = "Player"
        elif player_score < dealer_score:
            winner = "Dealer"
        else:
            winner = "Tie! No One Win"
    return winner
    
def hit_player(hand):
    global player_score
    card, _ = detect_card(frame) # Replace with actual card detection logic
    hand.append(card)
    player_score = calculate_hand_value(hand)

def stand_player():
    global game_over
    game_over = True

def reveal_dealer_cards(dealer_hand):
    # Reveal remaining dealer cards one by one
    for i in range(len(dealer_hand)):
        if dealer_hand[i] == "Closed Card":
            card, _ = detect_card(frame) # Replace with actual card detection logic
            dealer_hand[i] = card
            # Update dealer score and check for bust
            dealer_score = calculate_hand_value(dealer_hand)
            check_winner()
            # Show updated cards and score on UI
            draw_frame(frame, player_hand, dealer_hand, hit_button_area, stand_button_area, play_again_button_area, player_score, dealer_score, winner)
            # Add a delay between card reveals for dramatic effect (optional)
            
def play_again():
    # Reset game state variables
    global player_hand, dealer_hand, player_score, dealer_score, game_over, winner
    player_hand, dealer_hand = [], []
    player_score, dealer_score = 0, 0
    game_over, winner = False, None

# Main loop
vid = cv2.VideoCapture(2)
if not vid.isOpened():
    print("Camera not available")
    exit()
ret, frame = vid.read()    
player_hand = []
dealer_hand = []
player_score = 0
dealer_score = 0
winner = None
game_over = False
area = None
split_line = frame.shape[1] // 2
player_area = (0, split_line)

#loop
while True:
    ret, frame = vid.read()
    if not ret:
       print("Invalid frame")
       break

     # Preprocess the frame
    #processed_frame = preprocess_image(frame)

    # Detect cards in the preprocessed frame
    #detected_cards = detect_card(processed_frame)
    
    detected_result, detected_cards = detect_card(frame)

    predicted_card_names = [card[0] for card in detected_cards] if detected_cards else None

    #print("Detected cards:", predicted_card_names)
    player_hand, dealer_hand, player_score, dealer_score =  handle_detected_cards(detected_cards)
    
    #cv2.imshow("Detected Cards", predictions)
    player_score = calculate_hand_value(player_hand)
    dealer_score = calculate_hand_value(dealer_hand)
    cv2.putText(frame, f"Player Score: {player_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Dealer Score: {dealer_score}", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    check_winner(player_score,dealer_score)
    
    key = cv2.waitKey(1) & 0xFF            
    if key == ord('h'):  # Jika tombol 'h' ditekan (hit)
    # Display "Player Draw 1 Cards"
        cv2.putText(frame, "Player Draw 1 Cards", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        hit_player(player_hand)
        check_winner()
    elif key == ord('s'):  # Jika tombol 's' ditekan (stand)
        # Display "Dealer Reveal"
        cv2.putText(frame, "Dealer Reveal", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        stand_player()
    elif key == ord('y'):  # Jika tombol 'y' ditekan (play again)
        # Display "Starting new Game..."
        cv2.putText(frame, "Starting new Game...", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        play_again()

    # Update game state if game over
    if game_over:
        reveal_dealer_cards(dealer_hand)
        check_winner()
        if winner:
            winner_text = f"Winner: {winner}"
            cv2.putText(frame, winner_text, (frame.shape[1] // 2, frame.shape[0] // 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=3)
    
    cv2.putText(frame, f"Player Score: {player_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Dealer Score: {dealer_score}", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    player_score = calculate_hand_value(player_hand)
    dealer_score = calculate_hand_value(dealer_hand)
    winner = check_winner(player_score, dealer_score)
    # Draw frame with cards, buttons, and scores
    draw_frame(frame, player_hand, dealer_hand, player_score, dealer_score)

    # Show frame and check for key press to exit
    cv2.imshow("Blackjack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
vid.release()
cv2.destroyAllWindows()