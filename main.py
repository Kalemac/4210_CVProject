import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

game_piece = cv.imread('images/game_piece.jpg', cv.IMREAD_UNCHANGED)
game_piece90 = cv.imread('images/game_piece90.jpg', cv.IMREAD_UNCHANGED)
mid_game = cv.imread('images/game1.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(mid_game, game_piece, cv.TM_CCOEFF_NORMED)

cv.imshow('Result', result)
cv.waitKey()

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

threshold = 0.85
locations = np.where(result >= threshold)
locations = list(zip(*locations[::-1]))

if locations:
    print('Found Piece.')

    needle_w = game_piece.shape[1]
    needle_h = game_piece.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    # Loop over all the locations and draw their rectangle
    for loc in locations:
        # Determine the box positions
        top_left = loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
        # Draw the box
        cv.rectangle(mid_game, top_left, bottom_right, line_color, line_type)

    cv.imshow('Matches', mid_game)
    cv.waitKey()
    cv.imwrite('images/result.jpg', mid_game)

else:
    print('Piece not found.')
