const int INDEX = 0;
const int MIDDLE = 1;
const int RING = 1;
const int PINKY = 1;

const int motorPins[][3] = {
  {22, 23, 2}
};

const int encoderPins[][2] = {
  {25,24}
};

const int hallPins[] = {A0};

const int NUM_FINGERS = 5;

volatile long encoderValues[] = {0}; 
volatile int lastEncoded[] = {0}; 
int encoderOffsets[] = {};

const int homingCutOff1 = 620;
const int homingCutOff2 = 550;
bool initialization = true;

unsigned long lastPrint = 0;
long lastEncoderValue = 0;
float rpm = 0;
const int encoderPulsesPerRev = 14 * 150;

void stopMotor(int motorIdx){
  analogWrite(motorPins[motorIdx][2], 0);
  digitalWrite(motorPins[motorIdx][0], HIGH);
  digitalWrite(motorPins[motorIdx][1], LOW);
}


void homingSequence(){
  for (int i = 0; i < NUM_FINGERS; i++) {
     stopMotor(i);
     while (analogRead(hallPins[i]) > homingCutOff2){
      analogWrite(motorPins[i][2], 200);
      digitalWrite(motorPins[i][0], LOW);
      digitalWrite(motorPins[i][1], HIGH);
    }
    stopMotor(i);
    delay(50);
    while (analogRead(hallPins[i]) < homingCutOff1 ){
      analogWrite(motorPins[i][2], 200);
      digitalWrite(motorPins[i][0], HIGH);
      digitalWrite(motorPins[i][1], LOW);
    }
    stopMotor(i);
    delay(100);
    encoderOffsets[i] = encoderValues[i];
  }
}


void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  // Motor pins
  for (int j= 0; j < NUM_FINGERS; j ++){
    for (int i = 0; i < 3; i++) {  
      pinMode(motorPins[j][i], OUTPUT);
    }
    pinMode(hallPins[j], INPUT);
  }
  
  Serial.begin(9600);

  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(encoderPins[i][0], INPUT_PULLUP);
    pinMode(encoderPins[i][1], INPUT_PULLUP);
    int MSB = digitalRead(encoderPins[i][0]);
    int LSB = digitalRead(encoderPins[i][1]);
    lastEncoded[i] = (MSB << 1) | LSB;
  } 
  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][0]), updateEncoderIndex, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][1]), updateEncoderIndex, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[MIDDLE][0]), updateEncoderIndex, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPins[MIDDLE][1]), updateEncoderIndex, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[RING][0]), updateEncoderIndex, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPins[RING][1]), updateEncoderIndex, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][0]), updateEncoderIndex, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][1]), updateEncoderIndex, CHANGE);
}
void updateEncoderMiddle() {
  int MSB = digitalRead(encoderPins[MIDDLE][0]);
  int LSB = digitalRead(encoderPins[MIDDLE][1]);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded[MIDDLE] << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    encoderValues[MIDDLE]--;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    encoderValues[MIDDLE]++;

  lastEncoded[MIDDLE] = encoded;
}

void updateEncoderRing() {
  int MSB = digitalRead(encoderPins[RING][0]);
  int LSB = digitalRead(encoderPins[RING][1]);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded[RING] << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    encoderValues[RING]--;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    encoderValues[RING]++;

  lastEncoded[RING] = encoded;
}

void updateEncoderPinky() {
  int MSB = digitalRead(encoderPins[PINKY][0]);
  int LSB = digitalRead(encoderPins[PINKY][1]);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded[PINKY] << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    encoderValues[PINKY]--;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    encoderValues[PINKY]++;

  lastEncoded[PINKY] = encoded;
}

void updateEncoderIndex() {
  int MSB = digitalRead(encoderPins[INDEX][0]);
  int LSB = digitalRead(encoderPins[INDEX][1]);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded[INDEX] << 2) | encoded;

  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    encoderValues[INDEX]--;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    encoderValues[INDEX]++;

  lastEncoded[INDEX] = encoded;
}

void goUp(int motorIdx){
  digitalWrite(motorPins[motorIdx][0], LOW);
  digitalWrite(motorPins[motorIdx][1], HIGH);
}

void goDown(int motorIdx){
  digitalWrite(motorPins[motorIdx][0], HIGH);
  digitalWrite(motorPins[motorIdx][1], LOW);
}

void goToPosition(int motorIdx, int pos){
  int tolerance = 100;
  if (encoderValues[motorIdx] < pos){
    goUp(motorIdx);
  }
  if (encoderValues[motorIdx] >= pos){
    goDown(motorIdx);
  }
  while (abs(encoderValues[motorIdx] - pos) > tolerance ){
    analogWrite(motorPins[motorIdx][2], 200);
  }
  stopMotor(motorIdx);
}

void loop() {
  if (initialization){
    delay(3000);
    homingSequence();
    initialization = false;
    delay(3000);
  }
  goToPosition(0, 10000);
  delay(1000);
  goToPosition(0, 200);
  delay(1000);
  // for (int i = 100; i <=200; i++) {
  //   analogWrite(motorPins[0][2], i);
  //   digitalWrite(motorPins[0][0], LOW);
  //     digitalWrite(motorPins[0][1], HIGH);
  //   delay(5);
  //   Serial.println(indexEncoderValue-indexEncoderOffset);
  // }
  // stopMotor(0);
  // delay(1000);
  // for (int i = 200; i >= 100; i--) {
  //   analogWrite(motorPins[0][2], i);
  //   digitalWrite(motorPins[0][0], HIGH);
  //     digitalWrite(motorPins[0][1], LOW);
  //   delay(5);
  //   Serial.println(indexEncoderValue-indexEncoderOffset);
  // }
  // stopMotor(0);
  // delay(1000);

}

