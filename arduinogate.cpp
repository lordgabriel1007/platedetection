#include <Servo.h>

// Define servo pin
const int servoPin = 9;

// Define LED pins
const int RED = 2;
const int GREEN = 3;
const int YELLOW = 4;

// Create a servo object
Servo myServo;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Attach servo to the specified pin
  myServo.attach(servoPin);

  // Set LED pins as output
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(YELLOW, OUTPUT);

  // Initialize servo to 0 degrees
  myServo.write(0);
  delay(1000);
}

void loop() {
  // Check if there is data available on the serial port
  if (Serial.available() > 0) {
    // Read the incoming data
    char data = Serial.read();

    // Process the received data
    if (data == 'o') {
      // Open the gate
      myServo.write(150);
      digitalWrite(YELLOW, HIGH);
    } else if (data == 'c') {
      // Close the gate
      myServo.write(0);
      digitalWrite(YELLOW, LOW);
      digitalWrite(RED, LOW);
      digitalWrite(GREEN, LOW);
    } else if (data == 'r') {
      // Turn on the red LED
      digitalWrite(RED, HIGH);
    } else if (data == 'og') {
      // Open the gate and turn on the green LED
      myServo.write(150);
      digitalWrite(YELLOW, HIGH);
      digitalWrite(GREEN, HIGH);
    } else if (data == 'x') {
      // Stop the servo and clean up
      myServo.detach();
      digitalWrite(RED, LOW);
      digitalWrite(GREEN, LOW);
      digitalWrite(YELLOW, LOW);
      Serial.println("Servo stopped");
      exit(0);
    }
  }
}