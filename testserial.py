import serial

ser = None
try:
    ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the appropriate serial port
    while True:
        command = input("Enter command: ")
        if command == "exit":
            break
        ser.write(command.encode())
except Exception as e:
    print(f"Could not connect to Arduino - gate control. Will proceed without gate control. Error: {e}")
finally:
    if ser:
        ser.close()
