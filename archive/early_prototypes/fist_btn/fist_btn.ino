const int BUTTON1_PIN = 19;
const int BUTTON2_PIN = 21;

void setup() {
  Serial.begin(115200);

  pinMode(BUTTON1_PIN, INPUT_PULLUP);
  pinMode(BUTTON2_PIN, INPUT_PULLUP);
}

void loop() {
  int button1 = digitalRead(BUTTON1_PIN);
  int button2 = digitalRead(BUTTON2_PIN);

  if (button1 == LOW) {
    Serial.println("버튼 19 눌림");
  } else {
    Serial.println("버튼 19 안 눌림");
  }

  if (button2 == LOW) {
    Serial.println("버튼 21 눌림");
  } else {
    Serial.println("버튼 21 안 눌림");
  }

  Serial.println("----------------");

  delay(400);
}