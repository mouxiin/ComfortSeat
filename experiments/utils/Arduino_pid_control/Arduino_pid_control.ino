// ===== 控制引脚 =====
const uint8_t PWM_PIN_CH1 = 9;     // channel 1
const uint8_t PWM_PIN_CH2 = 10;    // channel 2

// 输入范围 0~30
const uint8_t MAX_CODE = 30;

bool replyEnabled = false;         // 串口回显

char lineBuf[32];
uint8_t lineLen = 0;

void applyCommand(int ch, int code); // forward

// ===== 串口解析 =====
void tryParseLine() {
  lineBuf[lineLen] = '\0';
  int ch = 0, code = 0;

  if (sscanf(lineBuf, "%d %d", &ch, &code) == 2) {
    applyCommand(ch, code);
  } else if (replyEnabled) {
    Serial.print("ERR: expect '1 20', got: ");
    Serial.println(lineBuf);
  }
}

void serialPumpOnce() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        tryParseLine();
        lineLen = 0;
      }
      continue;
    }

    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
      if (replyEnabled) Serial.println("ERR: line too long");
    }
  }
}

// ===== 将 0~30 映射到 0~255 PWM =====
uint8_t codeToPwm(int code) {
  if (code <= 0) return 0;
  if (code >= MAX_CODE) return 255;
  return (uint16_t)code * 255 / MAX_CODE;
}

// ===== Command处理 =====
void applyCommand(int ch, int code) {
  if (code < 0) code = 0;
  if (code > MAX_CODE) code = MAX_CODE;

  uint8_t pwmValue = codeToPwm(code);

  if (ch == 1) {
    // 开1号时关2号
    analogWrite(PWM_PIN_CH2, 0);
    analogWrite(PWM_PIN_CH2, 0);
    analogWrite(PWM_PIN_CH1, pwmValue);

  } else if (ch == 2) {
    // 开2号时关1号
    analogWrite(PWM_PIN_CH1, 0);
    analogWrite(PWM_PIN_CH1, 0);
    analogWrite(PWM_PIN_CH2, pwmValue);

  } else {
    // 非法通道，全部关闭
    analogWrite(PWM_PIN_CH1, 0);
    analogWrite(PWM_PIN_CH1, 0);
    analogWrite(PWM_PIN_CH2, 0);
  }

  if (replyEnabled) {
    Serial.print("OK ch=");
    Serial.print(ch);
    Serial.print(" code=");
    Serial.print(code);
    Serial.print(" pwm=");
    Serial.println(pwmValue);
  }
}

void setup() {
  Serial.begin(2000000);

  pinMode(PWM_PIN_CH1, OUTPUT);
  pinMode(PWM_PIN_CH2, OUTPUT);

  analogWrite(PWM_PIN_CH1, 0);
  analogWrite(PWM_PIN_CH2, 0);

  if (replyEnabled) {
    Serial.println("Ready: cmd '1 15' means CH1 PWM = 15/30");
  }
}

void loop() {
  serialPumpOnce();
}
