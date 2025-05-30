const int INDEX = 0;
const int MIDDLE = 1;
const int RING = 2;
const int PINKY = 3;

const int NUM_FINGERS = 4;

const int motorPins[NUM_FINGERS][3] = {
  {51, 50, 2}, // Index
  {47, 46, 3}, // Middle
  {43, 42, 4}, // Ring
  {37, 36, 5}, // Pinky
};

const int encoderPins[NUM_FINGERS][2] = {
  {53, 52}, // Index
  {49, 48}, // Middle
  {45, 44}, // Ring
  {39, 38}, // Pinky
};

const int hallPins[NUM_FINGERS] = {A0,A1,A2,A3};

volatile long t_rise[NUM_FINGERS] = {0,0,0,0};
volatile int directions[NUM_FINGERS] = {1,1,1,1};
volatile long encoderValues[NUM_FINGERS] = {0,0,0,0};
volatile int lastEncoded[NUM_FINGERS] = {0,0,0,0}; 
int encoderOffsets[NUM_FINGERS] = {0,0,0,0};

const int homingCutOff1[NUM_FINGERS] = {620, 610, 620, 620};
const int homingCutOff2[NUM_FINGERS] = {550, 550, 550, 550};
bool initialization = true;


int numActivations[NUM_FINGERS] = {0,0,0,0};
const int NUM_BEFORE_CALIBRATION = 20;


void stopMotor(int motorIdx){
  analogWrite(motorPins[motorIdx][2], 0);
  digitalWrite(motorPins[motorIdx][0], HIGH);
  digitalWrite(motorPins[motorIdx][1], LOW);
}


void homingSequence(int i){
  stopMotor(i);
  while (analogRead(hallPins[i]) > homingCutOff2[i]){
    Serial.print("Finger: ");
    Serial.print(i);
    Serial.print("; Hall: ");
    Serial.print(analogRead(hallPins[i]));
    Serial.print("; Encoder: ");
    Serial.println(encoderValues[i]);
    analogWrite(motorPins[i][2], 200);
    goUp(i);
  }
  stopMotor(i);
  delay(50);
  while (analogRead(hallPins[i]) < homingCutOff1[i]){
    Serial.print("Finger: ");
    Serial.print(i);
    Serial.print("; Hall: ");
    Serial.print(analogRead(hallPins[i]));
    Serial.print("; Encoder: ");
    Serial.println(encoderValues[i]);
    analogWrite(motorPins[i][2], 200);
    goDown(i);
  }
  stopMotor(i);
  delay(100);
  encoderOffsets[i] = encoderValues[i];
  Serial.print("Offset" );
  Serial.println(encoderOffsets[i]);
  
}


void setup() {
  Serial.begin(9600);
  // Motor pins
  for (int j= 0; j < NUM_FINGERS; j ++){
    for (int i = 0; i < 3; i++) {  
      pinMode(motorPins[j][i], OUTPUT);
    }
    pinMode(hallPins[j], INPUT);
    stopMotor(j);
  }
  


  for (int i = 0; i < NUM_FINGERS; i++) {
    pinMode(encoderPins[i][0], INPUT_PULLUP);
    pinMode(encoderPins[i][1], INPUT_PULLUP);
  } 
  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][0]), updateEncoderIndex, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[MIDDLE][0]), updateEncoderMiddle, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[RING][0]), updateEncoderRing, CHANGE);

  attachInterrupt(digitalPinToInterrupt(encoderPins[PINKY][0]), updateEncoderPinky, CHANGE);
}

void updateEncoderIndex(){
  int idx = INDEX;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
}
  
void updateEncoderMiddle(){
  int idx = MIDDLE;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
}

void updateEncoderRing(){
  int idx = RING;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
}

void updateEncoderPinky(){
  int idx = PINKY;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
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
  int tolerance = 20;
  if (encoderValues[motorIdx]-encoderOffsets[motorIdx] < pos){
    goUp(motorIdx);
  }
  if (encoderValues[motorIdx]-encoderOffsets[motorIdx] >= pos){
    goDown(motorIdx);
  }
  while (abs((encoderValues[motorIdx]-encoderOffsets[motorIdx]) - pos) > tolerance ){
    analogWrite(motorPins[motorIdx][2], 200);
    Serial.print("Finger:");
    Serial.print(motorIdx);
    Serial.print(", ");
    Serial.print("Encoder:");
    Serial.println(encoderValues[motorIdx]-encoderOffsets[motorIdx]);
  }
  stopMotor(motorIdx);
  numActivations[motorIdx]++;
}

void loop() {
  // analogWrite(motorPins[RING][2], 200);
  //   digitalWrite(motorPins[RING][0], LOW);
  //     digitalWrite(motorPins[RING][1], HIGH);
      // Serial.println(analogRead(hallPins[RING]));
  if (initialization){
    delay(3000);
    for (int i = 2; i < NUM_FINGERS; i ++){
      homingSequence(i);
    }
    initialization = false;
    delay(3000);
  }
  for (int i = 2; i < NUM_FINGERS; i ++){
    goToPosition(i, 4000); 
    delay(1000);
    goToPosition(i, 100);
    delay(1000);
  }
  // check if we need to recalibrate any fingers
  for (int i = 0; i < NUM_FINGERS; i ++){
    if (numActivations[i] >= NUM_BEFORE_CALIBRATION){
      homingSequence(i);
    }
  } 
}

