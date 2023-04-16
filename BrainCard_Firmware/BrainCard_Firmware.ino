#include <ADS1299.h>

#define PIN_DRDY 2  // Data Ready Pin
#define PIN_CS 9    // Chip Select Pin
#define PIN_RESET 8 // Reset Pin
#define PIN_START 7 // Start Pin

ADS1299 ads;

// Initialize the ADS1299 library with the specified pin configuration

void setup() {
  // Set up the serial connection
  Serial.begin(115200);
  Serial.println("Starting");
  
  ads.setup(PIN_DRDY,PIN_CS);
  delay(20);

  ads.RESET();
}

void loop() {

}
