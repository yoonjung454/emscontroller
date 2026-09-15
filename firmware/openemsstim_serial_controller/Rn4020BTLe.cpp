/**
 * ArduinoSoftware_Arduino_IDE
 *
 *  Copyright 2016 by Tim Duente <tim.duente@hci.uni-hannover.de>
 *  Copyright 2016 by Max Pfeiffer <max.pfeiffer@hci.uni-hannover.de>
 *
 *  Licensed under "The MIT License (MIT) - military use of this product is forbidden - V 0.2".
 *  Some rights reserved. See LICENSE.
 *
 * @license "The MIT License (MIT) - military use of this product is forbidden - V 0.2"
 * <https://bitbucket.org/MaxPfeiffer/letyourbodymove/wiki/Home/License>
 */

/**
 * Minor Revisions to this file by Pedro Lopes <plopesresearch@gmail.com>, all credit remains with the original authors (see above).
 */

/*
 * Rn4020BTLe.cpp
 *
 *  Created on: 19.03.2015
 *      Author: Tim Duente
 */

#include "Rn4020BTLe.h"

Rn4020BTLe::Rn4020BTLe(uint8_t HW_Wake_Up, AltSoftSerial *serial) {
	this->HW_Wake_Up = HW_Wake_Up;
	pinMode(HW_Wake_Up, OUTPUT);
	digitalWrite(HW_Wake_Up, HIGH);
	this->serial = serial;
}

Rn4020BTLe::~Rn4020BTLe() {
	// TODO Auto-generated destructor stub
}

void Rn4020BTLe::reset() {
	//Alternate the HW_Wake_UP Pin of the RN4020 to reset it to factory settings
	digitalWrite(HW_Wake_Up, HIGH);
	delay(200);
	digitalWrite(HW_Wake_Up, LOW);
	delay(200);
	digitalWrite(HW_Wake_Up, HIGH);
	delay(200);
	digitalWrite(HW_Wake_Up, LOW);
	delay(200);
	digitalWrite(HW_Wake_Up, HIGH);
	delay(200);
	digitalWrite(HW_Wake_Up, LOW);
	delay(200);
	digitalWrite(HW_Wake_Up, HIGH);
	delay(3000);
}

bool Rn4020BTLe::check_baudrate(unsigned long baudrate){
    debug_print(F("\t\tCheck baudrate:"));
    String baud = String(baudrate);
    debug_println(baud);
    serial->begin(baudrate);
    delay(50);
    //Send a dummy command. Might fail. Next command should work properly
    serial->println("V");
    delay(50);
    debug_print(F("\t\tVersion: "));
    print_response();
  
    serial->println("SB,0");
    delay(150);
    debug_print(F("\t\tSet baudrate: "));
    bool check_baud =  print_response();

    if(!check_baud){
      return false;
    }
    serial->println(F("R,1")); //Reboot
    delay(100);
    debug_print(F("\t\tReboot?: "));
    print_response();

    return check_baud;
}

void Rn4020BTLe::init(String bluetoothName) {

  // Try finding the last set baudrate to initialize the bluetooth chip
  if(!check_baudrate(2400) && !check_baudrate(9600) && !check_baudrate(19200) && !check_baudrate(38400) && !check_baudrate(115200)){
    debug_println(F("No connection!"));
    return;
  }

	delay(1500);
	serial->begin(2400);

	bluetoothName = "SN," + bluetoothName;
	// Send a dummy command. Might fail. Next command should work properly
	serial->println("V");
	delay(100);
	debug_print(F("\t\tVersion: "));
	print_response();

	serial->println(bluetoothName);
	delay(100);
	debug_print(F("\t\tSet Name: "));
	print_response();

	// Set the RN4020 to Apple Bluetooth Accessory Design Guidelines mode RS 0x00004000

	/*
	 serial->println("SR,00004000");
	 delay(200);
	 printer(F("\t\tSet peripheral mode Apple Bluetooth Accessory Design Guidelines : "));
	 print_response();
	 */

	//Sets RN4020 in peripheral Mode with Auto Advertising
	serial->println(F("SR,20000000"));
	delay(100);
	debug_print(F("\t\tSet peripheral mode: "));
	print_response();

	//Enables private services.
	serial->println(F("SS,C0000001"));
	delay(100);
	debug_print(F("\t\tEnable private services: "));
	print_response();

	serial->println("PZ"); // Clear the current private service and characteristics
	delay(100);
	debug_print(F("\t\tClear private services: "));
	print_response();

	serial->println(F("PS,454d532d536572766963652d424c4531")); //  "PS,454d532d536572766963652d424c4531" in ASCII "EMS-Service-BLE1"
	delay(250);
	debug_print(F("\t\tSet new private service: "));
	print_response();

	serial->println(F("PC,454d532d537465756572756e672d4348,18,20")); // "PC,454d532d537465756572756e672d4348,18,20" in ASCII "EMS-SteuerungCH1" gets Handle 001C (proof with LS)
	delay(250);
	debug_print(F("\t\tSet new private service value: "));
	print_response();

	serial->println(F("SB,3"));
	delay(150);
	debug_print(F("\t\tSet baudrate: "));
	print_response();

	serial->println(F("R,1"));
	delay(100);
	debug_print(F("\t\tReboot?: "));
	print_response();

	start_communication();
	delay(1000);
}

void Rn4020BTLe::start_communication() {
	serial->begin(38400);
}

bool Rn4020BTLe::print_response() {
	String response;
	if (serial->available() > 0) {
		response = serial->readStringUntil('\n');
		response.trim();
		debug_println(response);
	} else {
		debug_println(F("No respond. Check baudrate."));
	}
  if(response.equals("AOK")){
    return true;
  }
  else{
    return false;
  }
 
}
