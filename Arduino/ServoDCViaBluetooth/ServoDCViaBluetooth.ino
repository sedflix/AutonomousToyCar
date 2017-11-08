#include <SoftwareSerial.h>// import the serial library




#include <Servo.h>

Servo servo;
int  servoPin = 9;
int motorPin1 = 6;
int motorPin2 = 5;
int motorPinEN = 10;
String str = "";
char mode;

void setup() {

  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorPinEN, OUTPUT);
  digitalWrite(motorPinEN, HIGH);
//  Genotronex.begin(9600);
  Serial.begin(9600);
  servo.attach(servoPin);
  brake();


}

void loop() {
  if (Serial.available()) {
    char x = char(Serial.read());

    if (x == 'a') {

//      Serial.println("in a");



      str = "";
      int i  = 0;
      int j = 0;
      
      while (true) {
        if (Serial.available())
        {
          str = str + "" + char(Serial.read());
          i++;
        }
        j ++;
        if (i >= 3 && j > 100) {
          break;
        }
      }

      int angle  = (atoi(str.c_str()));
//      Serial.println(angle);


      if (angle >= 0 && angle <= 180) {
        servo.write(angle);
      }
      else {
        servo.write(0);
      }


    }
    else if (x == 's') {

//      Serial.println("in s");

      while (true) {
        if (Serial.available()) {
          mode = char(Serial.read());
          break;
        }
      }

//      Serial.println(mode);

      if (mode == '1')
      {
//        Serial.println("forward");
        forward();
      }
      else if (mode == '2')
      {
//        Serial.println("backward");
        backward();
      }
      else if (mode == '3')
      {
        brake();
      }

    }//s code ends

  } // if available code

}

void forward() {
  digitalWrite(motorPinEN, HIGH);
  digitalWrite(motorPin1, LOW) ;
  digitalWrite(motorPin2, HIGH) ;
}

void backward() {
  digitalWrite(motorPinEN, HIGH);
  digitalWrite(motorPin1, HIGH) ;
  digitalWrite(motorPin2, LOW) ;
}

void brake() {
  digitalWrite(motorPinEN, LOW);
  digitalWrite(motorPin1, HIGH) ;
  digitalWrite(motorPin2, HIGH) ;
}

