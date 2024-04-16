import cv2

# Initialize a list to store click coordinates
click_positions = []
click_id = 1

def click_event(event, x, y, flags, param):
    global click_positions, click_id
    if event == cv2.EVENT_LBUTTONDOWN:
        click_positions.append((click_id, (x, y)))
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("image", img)
        click_id += 1

def main():
    global img
    # Load an image
    path = "/work/mbergst/TDT4265_Project/pitch.png"
    img = cv2.imread(path)

    if img is None:
        print("Error: Image not found.")
        return

    # Display the image
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)



    # Wait for the specified number of clicks
    while len(click_positions) < 36:
        if cv2.waitKey(20) & 0xFF == 27:
            break

    # Print the results
    for id, pos in click_positions:
        print(f"ID {id}: Position {pos}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
