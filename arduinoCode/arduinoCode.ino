const int INDEX = 0;
const int MIDDLE = 1;
const int RING = 2;
const int PINKY = 3;
const int THUMB_BASE_ROT = 4;
const int THUMB_BASE_EXT = 5;
const int THUMB = 6;

const int downButton = 12;
int downButtonState = 0;
const int upButton = 11;
int upButtonState = 0;
const int nextButton = 13;
int nextButtonState = 0;

const int NUM_FINGERS = 7;

const int motorPins[NUM_FINGERS][3] = {
  {51, 50, 2}, // Index
  {47, 46, 3}, // Middle
  {43, 42, 4}, // Ring
  {37, 36, 5}, // Pinky
  {33, 32, 6}, // Thumb Base Rotation
  {28, 29, 7}, // Thumb Base Extension
  {25, 24, 8}, // Thumb 
};

const int encoderPins[NUM_FINGERS][2] = {
  {53, 52}, // Index
  {48, 49}, // Middle
  {45, 44}, // Ring
  {39, 38}, // Pinky
  {35, 34}, // Thumb Base Rotation
  {31, 30}, // Thumb Base Extension
  {27, 26}, // Thumb 
};


const int MOTOR_SPEED = 225;
const int hallPins[NUM_FINGERS] = {A0,A1,A2,A3,A4,A5,A6};

int OPEN_POS[NUM_FINGERS] = {5000,4250,4500,4500, 1000, 1000, 2500};
int CLOSE_POS[NUM_FINGERS] = {500,500,500,500, 500, 500, 500};


volatile long t_rise[NUM_FINGERS] = {0,0,0,0,0,0,0};
volatile int directions[NUM_FINGERS] = {1,1,1,1,1,1,1};
volatile long encoderValues[NUM_FINGERS] = {0,0,0,0,0,0,0};
volatile int lastEncoded[NUM_FINGERS] = {0,0,0,0,0,0,0}; 
int encoderOffsets[NUM_FINGERS] = {0,0,0,0,0,0,0};

const int homingCutOff1[NUM_FINGERS] = {570, 580, 580, 580, 580, 580, 580};
const int homingCutOff2[NUM_FINGERS] = {550, 550, 550, 550, 550, 550, 550};
long timeOuts[NUM_FINGERS] = {3000,3000, 3000, 3000, 3000, 3000, 3000};
bool initialization = true;


int numActivations[NUM_FINGERS] = {0,0,0,0,0,0,0};
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
    analogWrite(motorPins[i][2], MOTOR_SPEED);
    goUp(i);
  }
  stopMotor(i);
  delay(50);
  int counter = 0;
  while (counter < 6){
    Serial.print("Finger: ");
    Serial.print(i);
    Serial.print("; Hall: ");
    Serial.print(analogRead(hallPins[i]));
    Serial.print("; Encoder: ");
    Serial.println(encoderValues[i]);
    analogWrite(motorPins[i][2], MOTOR_SPEED);
    goDown(i);
    if (analogRead(hallPins[i]) >= homingCutOff1[i]){
      counter ++;
    }
    else{
      counter = 0;
    }
  }
  stopMotor(i);
  delay(100);
  encoderOffsets[i] = encoderValues[i];
  Serial.print("Offset" );
  Serial.println(encoderOffsets[i]);
}


void setup() {
  Serial.begin(9600);

  // Homing buttons
  pinMode(downButton, INPUT);
  pinMode(upButton, INPUT);
  pinMode(nextButton, INPUT);

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
  attachInterrupt(digitalPinToInterrupt(encoderPins[INDEX][0]), updateEncoderIndex, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[MIDDLE][0]), updateEncoderMiddle, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[RING][0]), updateEncoderRing, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[PINKY][0]), updateEncoderPinky, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[THUMB_BASE_ROT][0]), updateEncoderThumbBaseRot, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[THUMB_BASE_EXT][0]), updateEncoderThumbBaseExt, RISING);

  attachInterrupt(digitalPinToInterrupt(encoderPins[THUMB][0]), updateEncoderThumb, RISING);
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

void updateEncoderThumbBaseRot(){
  int idx = THUMB_BASE_ROT;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
}

void updateEncoderThumbBaseExt(){
  int idx = THUMB_BASE_EXT;
  if (digitalRead(encoderPins[idx][1])){
    directions[idx] = -1;
  }
  else{
    directions[idx] = 1;
  }
  encoderValues[idx] += directions[idx];
}

void updateEncoderThumb(){
  int idx = THUMB;
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
    analogWrite(motorPins[motorIdx][2], MOTOR_SPEED);
    Serial.print("Finger:");
    Serial.print(motorIdx);
    Serial.print(", ");
    Serial.print("Encoder:");
    Serial.println(encoderValues[motorIdx]-encoderOffsets[motorIdx]);
  }
  stopMotor(motorIdx);
  numActivations[motorIdx]++;
}

void setAllFingers(int targetPos[]){
  int done[NUM_FINGERS] = {0,0,0,0,0,0,0};
  int tolerance = 20;
  long t0[] = {millis(), millis(), millis(), millis(), millis(), millis(), millis()};
  while (done[0]!= 1 || done[1]!= 1 || done[2]!= 1 || done[3]!= 1 || done[4]!= 1 || done[5] != 1 || done[6] != 1){
    Serial.print("0:");
    Serial.print(encoderValues[0]-encoderOffsets[0]);
    Serial.print(", ");
    Serial.print("1:");
    Serial.print(encoderValues[1]-encoderOffsets[1]);
    Serial.print(", ");
    Serial.print("2:");
    Serial.print(encoderValues[2]-encoderOffsets[2]);
    Serial.print(", ");
    Serial.print("3:");
    Serial.print(encoderValues[3]-encoderOffsets[3]);
    Serial.print(", ");
    Serial.print("4:");
    Serial.print(encoderValues[4]-encoderOffsets[4]);
    Serial.print(", ");
    Serial.print("5:");
    Serial.print(encoderValues[5]-encoderOffsets[5]);
     Serial.print(", ");
    Serial.print("6:");
    Serial.println(encoderValues[6]-encoderOffsets[6]);



    for (int i = 0; i < NUM_FINGERS; i ++){
      int pos = targetPos[i];
      if (encoderValues[i]-encoderOffsets[i] < pos){
        goUp(i);
      }
      else if (encoderValues[i]-encoderOffsets[i] >= pos){
        goDown(i);
      }

      if (abs((encoderValues[i]-encoderOffsets[i]) - pos) <= tolerance || millis()-t0[i] > timeOuts[i]){
        stopMotor(i);
        done[i] = 1;
      }
      else{
        analogWrite(motorPins[i][2], MOTOR_SPEED);
        done[i] = 0;
      }
    }
  }
  for (int i = 0; i < NUM_FINGERS; i ++){
     stopMotor(i);
  }
}

void pinch(){

  int phase1[NUM_FINGERS] = {5000,4250,4500,4500, 1800, 0, 2500};
  int phase2[NUM_FINGERS] = {5000,4250,4500,4500, 1800, 0, 2500};
  int phase3[NUM_FINGERS] = {2500,4250,4500,4500, 1800, 0, 2500};
  setAllFingers(phase1);
  delay(50);
  setAllFingers(phase2);
  delay(1000);
  setAllFingers(phase3);
}

void loop() {
  if (initialization){
    delay(3000);
    goUp(INDEX);
    goUp(MIDDLE);
    analogWrite(motorPins[INDEX][2], MOTOR_SPEED);
    analogWrite(motorPins[MIDDLE][2], MOTOR_SPEED);
    delay(500);
    stopMotor(INDEX);
    stopMotor(MIDDLE);

    while (digitalRead(nextButton) == 0){
      if (digitalRead(downButton)== 1){
        goDown(THUMB_BASE_EXT);
        analogWrite(motorPins[THUMB_BASE_EXT][2], MOTOR_SPEED);
      }
      else if (digitalRead(upButton) == 1){
        goUp(THUMB_BASE_EXT);
        analogWrite(motorPins[THUMB_BASE_EXT][2], MOTOR_SPEED);
      }
      else{
        stopMotor(THUMB_BASE_EXT);
      }
    }
    delay(500);
    while (digitalRead(nextButton) == LOW){
      if (digitalRead(downButton)== HIGH){
        goDown(THUMB_BASE_ROT);
        analogWrite(motorPins[THUMB_BASE_ROT][2], MOTOR_SPEED);
      }
      else if (digitalRead(upButton) == HIGH){
        goUp(THUMB_BASE_ROT);
        analogWrite(motorPins[THUMB_BASE_ROT][2], MOTOR_SPEED);
      }
      else{
        stopMotor(THUMB_BASE_ROT);
      }
    }
    
    homingSequence(4);
    homingSequence(5);
    homingSequence(6);
    goToPosition(THUMB, OPEN_POS[THUMB]);
   
    goToPosition(6, OPEN_POS[6]);
    for (int i = 0; i < NUM_FINGERS; i ++){
      if (i == 5) {
        goToPosition(INDEX, OPEN_POS[INDEX]);
        goToPosition(MIDDLE, OPEN_POS[MIDDLE]);
      }
      homingSequence(i);
    }
    initialization = false;
    delay(3000);
  }

  pinch();
  delay(5000);

  // setAllFingers(OPEN_POS);
  // delay(3000);
  // setAllFingers(CLOSE_POS);
  // delay(3000);
  // for (int i = 0; i < NUM_FINGERS; i ++){
  //   goToPosition(i, OPEN_POS[i]); 
  //   delay(1000);
  //   goToPosition(i, CLOSE_POS[i]);
  //   delay(1000);
  // }
  // check if we need to recalibrate any fingers

  // for (int i = 0; i < NUM_FINGERS; i ++){
  //   if (numActivations[i] >= NUM_BEFORE_CALIBRATION){
  //     homingSequence(i);
  //   }
  // } 
}

