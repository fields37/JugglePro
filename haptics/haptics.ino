#include <Wire.h>
#include "Adafruit_DRV2605.h"

#define TCAADDR 0x70
#define LEFT 7
#define RIGHT 0
Adafruit_DRV2605 drv;
int data;

// selects the given i2c bus in the i2c multiplexer
void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

// sends a vibrate signal to the i2c device in the given i2c bus
void vibrate(uint8_t i) {
  tcaselect(i);
  drv.selectLibrary(1);
  drv.setWaveform(0,14);
  drv.setWaveform(1,0);
  drv.setMode(DRV2605_MODE_INTTRIG); // I2C trigger by sending 'go' command 
//  drv.useERM();
  drv.go();
}

void setup() {
  Serial.begin(115200);
  tcaselect(LEFT);
  tcaselect(RIGHT);
  drv.begin();
  data = 0;
}

void loop() {
  if (Serial.available()) {
    data = Serial.read();
    if (data == '1') {
      // play the effect on the left hand
      vibrate(LEFT);
      delay(500);
    } else if (data == '2') {
      // play the effect on the right hand
      vibrate(RIGHT);
      delay(500);
    }
  }

//  // Haptics Debugging
//  vibrate(LEFT);
//  delay(500);
//  vibrate(RIGHT);
//  delay(1000);
}
