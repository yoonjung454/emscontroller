/**
 * openemsstim_serial_controller.ino
 * ----------------------------------
 * PATCH of the official arduino-openEMSstim.ino
 * (https://github.com/PedroLopes/openEMSstim/tree/master/arduino-openEMSstim)
 * that adds a small text-based USB-serial command set (PING/ARM/SET/STOP/
 * STOP_ALL/STATUS) used by serial_link.py, WITHOUT changing any pin wiring,
 * without replacing EMSChannel/EMSSystem/AD5252, and without deleting the
 * original Bluetooth/HEX/single-char command paths (they are still present,
 * just gated behind ENABLE_BLUETOOTH_MODULE in hardware_config.h, which
 * defaults to 0 because this project's prelim explicitly does not use BLE).
 *
 * You still need the original library files sitting next to this .ino,
 * UNCHANGED, exactly as they came from the official repo:
 *   AD5252.cpp / AD5252.h
 *   EMSChannel.cpp / EMSChannel.h
 *   EMSSystem.cpp / EMSSystem.h
 *   AltSoftSerial.cpp / AltSoftSerial.h / AltSoftSerial_Boards.h / AltSoftSerial_Timers.h
 *   Rn4020BTLe.cpp / Rn4020BTLe.h
 *   known_boards.h / known_timers.h
 * (only needed if you set ENABLE_BLUETOOTH_MODULE to 1 -- see hardware_config.h)
 *
 * New USB serial protocol (see README.md for the full spec):
 *   PC -> Arduino : PING | ARM,<ch> | SET,<ch>,<intensity0-100>,<duration_ms> |
 *                   STOP,<ch> | STOP_ALL | STATUS
 *   Arduino -> PC : PONG | ARMED,<ch> | OK,<ch> | STOPPED,<ch> | STATUS,... |
 *                   ERROR,<reason>
 *
 * @license "The MIT License (MIT) - military use of this product is forbidden - V 0.2"
 *  (same license as the original openEMSstim firmware this file patches)
 */

#include "Arduino.h"
#include "hardware_config.h"

#if ENABLE_BLUETOOTH_MODULE
#include "AltSoftSerial.h"
#endif
#include "Wire.h"
#include "AD5252.h"
#if ENABLE_BLUETOOTH_MODULE
#include "Rn4020BTLe.h"
#endif
#include "EMSSystem.h"
#include "EMSChannel.h"
#include "avr/pgmspace.h"

//BT: the string below is how your EMS module will show up for other BLE devices
#define EMS_BLUETOOTH_ID "TEST1"

//DEBUG: setup for verbose mode (prints debug messages if DEBUG_ON is 1)
#define DEBUG_ON 1

//USB: allows commands using the full original protocol (refer to https://github.com/PedroLopes/openEMSstim)
//     kept for reference / BLE-bridge use, OFF by default so it doesn't fight our simple protocol below.
#define USB_FULL_COMMANDS_ACTIVE 0

//USB: allows the original one-char test commands (1,2,q,a,w,s) for quick relay/poti testing.
#define USB_TEST_COMMANDS_ACTIVE 0

//USB: our new PING/ARM/SET/STOP/STOP_ALL/STATUS text protocol used by serial_link.py. ON by default.
#define USB_SIMPLE_PROTOCOL_ACTIVE 1

//helper print function that handles the DEBUG_ON flag automatically
void printer(String msg, boolean force = false) {
  if (DEBUG_ON || force) {
    Serial.println(msg);
  }
}

//Initialization of control objects
//NOTE: pin numbers below are the official openEMSstim example wiring -- see
//hardware_config.h for what to double-check before trusting them on your board.
#if ENABLE_BLUETOOTH_MODULE
AltSoftSerial softSerial;
#endif
AD5252 digitalPot(0);
#if ENABLE_BLUETOOTH_MODULE
Rn4020BTLe bluetoothModule(2, &softSerial);
#endif
EMSChannel emsChannel1(5, 4, A2, &digitalPot, 1);
EMSChannel emsChannel2(6, 7, A3, &digitalPot, 3);
EMSSystem emsSystem(2);

// ---- our patch: simple-protocol state --------------------------------------
bool pcArmed = false;
unsigned long lastHeartbeatPc = 0;

void setup() {
	Serial.begin(SERIAL_BAUD_RATE);
#if ENABLE_BLUETOOTH_MODULE
	softSerial.setTimeout(100);
#endif
	Serial.setTimeout(50);
	printer("\nSETUP:");
	Serial.flush();

#if ENABLE_BLUETOOTH_MODULE
	//Reset and Initialize the Bluetooth module
	printer("\tBT: RESETTING");
	bluetoothModule.reset();
	printer("\tBT: RESET DONE");
	printer("\tBT: INITIALIZING");
	bluetoothModule.init(EMS_BLUETOOTH_ID);
	printer("\tBT: INITIALIZED");
#else
	printer("\tBT: DISABLED (ENABLE_BLUETOOTH_MODULE=0 in hardware_config.h)");
#endif

	//Add the EMS channels and start the control
	printer("\tEMS: INITIALIZING CHANNELS");
	emsSystem.addChannelToSystem(&emsChannel1);
	emsSystem.addChannelToSystem(&emsChannel2);
	EMSSystem::start();
	printer("\tEMS: INITIALIZED");
	printer("\tEMS: STARTED");

	// Safety: every channel starts deactivated (EMSChannel's own constructor
	// already does this via digitalWrite(...LOW), this is an explicit
	// belt-and-suspenders re-assertion).
	emsSystem.shutDown();
	pcArmed = false;

	printer("SETUP DONE (LED 13 WILL BE ON)");
	pinMode(13, OUTPUT);
	digitalWrite(13, HIGH);

	lastHeartbeatPc = millis();
	printer("\tSIMPLE PROTOCOL READY (PING/ARM/SET/STOP/STOP_ALL/STATUS)", true);
}

String command = "";
String hexCommandString;
const String BTLE_DISCONNECT = "Connection End";

void loop() {

#if ENABLE_BLUETOOTH_MODULE
	if (softSerial.available() > 0) {
		String message = softSerial.readStringUntil('\n');
		message.trim();
                printer("\tBT: received command: " + String(message));
		processMessage(message);
                softSerial.flush(); //never tested
	}
#endif

	//Checks whether a signal has to be stoped (millis()-based, non-blocking)
	if (emsSystem.check() > 0) {

	}

	//Communicate to the EMS-module over USB
        if (Serial.available() > 0) {
           if (USB_SIMPLE_PROTOCOL_ACTIVE) {
             String message = Serial.readStringUntil('\n');
             message.trim();
             handleSimpleCommand(message);
           } else if (USB_FULL_COMMANDS_ACTIVE) {
             String message = Serial.readStringUntil('\n');
             printer("\tUSB: received command: " + String(message));
             message.trim();
	     processMessage(message);
           } else if (USB_TEST_COMMANDS_ACTIVE) {
             char c = Serial.read();
             printer("\tUSB-TEST-MODE: received command: " + char(c));
	     doCommand(c);
	   }
          Serial.flush();
	}

	// ---- our patch: PC heartbeat watchdog --------------------------------
	checkHeartbeatWatchdog();
}

// ============================================================================
// ---- our patch: simple PING/ARM/SET/STOP/STOP_ALL/STATUS protocol --------
// ============================================================================

bool isChannelSupported(int channel) {
	if (channel == 1) return CHANNEL_1_ENABLED;
	if (channel == 2) return CHANNEL_2_ENABLED;
	return false;
}

EMSChannel* channelObject(int channel) {
	if (channel == 1) return &emsChannel1;
	if (channel == 2) return &emsChannel2;
	return NULL;
}

void checkHeartbeatWatchdog() {
	if (pcArmed && (millis() - lastHeartbeatPc > (unsigned long) HEARTBEAT_TIMEOUT_MS)) {
		emsSystem.shutDown();
		pcArmed = false;
		printer("\tWATCHDOG: PC heartbeat timeout -> all channels deactivated", true);
	}
}

void handleSimpleCommand(String message) {
	if (message.length() == 0) return;

	// Any well-formed line from the PC counts as a sign of life, not just
	// PING -- a PC that is busy sending SET commands every 500ms should not
	// be treated as silent just because it isn't also sending PING.
	lastHeartbeatPc = millis();

	if (message == "PING") {
		Serial.println("PONG");
		return;
	}

	if (message == "STOP_ALL") {
		emsSystem.shutDown();
		pcArmed = false;
		Serial.println("STOPPED,ALL");
		return;
	}

	if (message == "STATUS") {
		Serial.print("STATUS,armed=");
		Serial.print(pcArmed ? "1" : "0");
		Serial.print(",ch1_active=");
		Serial.print(emsChannel1.isActivated() ? "1" : "0");
		Serial.print(",ch1_intensity=");
		Serial.print(emsChannel1.getIntensity());
		Serial.print(",ch2_supported=");
		Serial.println(CHANNEL_2_ENABLED ? "1" : "0");
		return;
	}

	int firstComma = message.indexOf(',');
	String verb = (firstComma == -1) ? message : message.substring(0, firstComma);

	if (verb == "ARM") {
		int channel = (firstComma == -1) ? -1 : message.substring(firstComma + 1).toInt();
		if (!isChannelSupported(channel)) {
			Serial.print("ERROR,UNSUPPORTED_CHANNEL,");
			Serial.println(channel);
			return;
		}
		pcArmed = true;
		Serial.print("ARMED,");
		Serial.println(channel);
		return;
	}

	if (verb == "SET") {
		// SET,<channel>,<intensity 0-100>,<duration_ms>
		int c1 = message.indexOf(',', firstComma + 1);
		int c2 = (c1 == -1) ? -1 : message.indexOf(',', c1 + 1);
		if (firstComma == -1 || c1 == -1 || c2 == -1) {
			Serial.println("ERROR,BAD_FORMAT");
			return;
		}

		int channel = message.substring(firstComma + 1, c1).toInt();
		int intensity = message.substring(c1 + 1, c2).toInt();
		long duration = message.substring(c2 + 1).toInt();

		if (!pcArmed) {
			Serial.println("ERROR,NOT_ARMED");
			return;
		}
		if (!isChannelSupported(channel)) {
			Serial.print("ERROR,UNSUPPORTED_CHANNEL,");
			Serial.println(channel);
			return;
		}
		if (intensity < 0 || intensity > 100) {
			Serial.println("ERROR,INTENSITY_RANGE");
			return;
		}
		if (duration < 0) duration = 0;
		if (duration > (long) MAX_COMMAND_TTL_MS) duration = MAX_COMMAND_TTL_MS;

		EMSChannel* target = channelObject(channel);
		target->setIntensity(intensity);
		target->setSignalLength((int) duration);
		target->activate();
		target->applySignal(); // sets the millis()-based auto-deactivate deadline; emsSystem.check() in loop() enforces it

		Serial.print("OK,");
		Serial.println(channel);
		return;
	}

	if (verb == "STOP") {
		int channel = (firstComma == -1) ? -1 : message.substring(firstComma + 1).toInt();
		if (!isChannelSupported(channel)) {
			Serial.print("ERROR,UNSUPPORTED_CHANNEL,");
			Serial.println(channel);
			return;
		}
		channelObject(channel)->deactivate();
		Serial.print("STOPPED,");
		Serial.println(channel);
		return;
	}

	Serial.print("ERROR,UNKNOWN_COMMAND,");
	Serial.println(message);
}

// ============================================================================
// ---- original openEMSstim code below, UNCHANGED ---------------------------
// (HEX/BLE command path + single-char test commands. Only reachable if you
//  re-enable ENABLE_BLUETOOTH_MODULE and/or USB_FULL_COMMANDS_ACTIVE /
//  USB_TEST_COMMANDS_ACTIVE above.)
// ============================================================================

//Convert-functions for HEX-Strings "4D"->"M"
char convertToHexCharsToOneByte(char one, char two) {
	char byteOne = convertHexCharToByte(one);
	char byteTwo = convertHexCharToByte(two);
	if (byteOne != -1 && byteTwo != -1)
		return byteOne * 16 + byteTwo;
	else {
		return -1;
	}
}

char convertHexCharToByte(char hexChar) {
	if (hexChar >= 'A' && hexChar <= 'F') {
		return hexChar - 'A';
	} else if (hexChar >= '0' && hexChar <= '9') {
		return hexChar - '0';
	} else {
		return -1;
	}
}

const char ems_channel_1_active[] PROGMEM =    "\tEMS: Channel 1 active";
const char ems_channel_1_inactive[]  PROGMEM = "\tEMS: Channel 1 inactive";
const char ems_channel_2_active[] PROGMEM =    "\tEMS: Channel 2 active";
const char ems_channel_2_inactive[] PROGMEM =  "\tEMS: Channel 2 inactive";
const char ems_channel_1_intensity[] PROGMEM = "\tEMS: Intensity Channel 1: ";
const char ems_channel_2_intensity[] PROGMEM = "\tEMS: Intensity Channel 2: ";

const char* const string_table_outputs[] PROGMEM = {ems_channel_1_active, ems_channel_1_inactive, ems_channel_2_active, ems_channel_2_inactive, ems_channel_1_intensity, ems_channel_2_intensity};

char buffer[32];

//process a command message (according to protocol, check github for that)
void processMessage(String message) {
  if (message.charAt(0) == 'W' && message.charAt(1) == 'V') {
    int lastIndexOfComma = message.lastIndexOf(',');
    hexCommandString = message.substring(lastIndexOfComma + 1,
    message.length() - 1);
    command = "";
    printer("\tEMS_CMD: HEX command length: ");
    printer(String(hexCommandString.length()));
    printer(hexCommandString);
    for (unsigned int i = 0; i < hexCommandString.length(); i = i + 2) {
      char nextChar = convertToHexCharsToOneByte(hexCommandString.charAt(i),hexCommandString.charAt(i + 1));
      command = command + nextChar;
    }
    printer("\tEMS_CMD: Converted HEX command: ");
    printer(command);
    emsSystem.doCommand(&command);
  } else if (message.equals(BTLE_DISCONNECT)) {
    printer("\tBT: Disconnected");
    emsSystem.shutDown();
  }
  else {
    printer("\tCommand NON HEX:");
    printer(message);
    doCommand(message[0]);
  }
}



// TESTING COMMANDS
//   For quick testing (e.g., opening the optocoupler ports, etc) you can use these single-char commands (one command = one char).
//   The cavailable commands are:
//     "1": toggles the channel 1 between open/closed state (when the channel is closed, the potenciometer position is reset to minimum, i.e., 255)
//     "2": toggles the channel 1 between open/closed state (when the channel is closed, the potenciometer position is reset to minimum, i.e., 255)
//     "q": increases EMS signal on channel 1 by decreasing the digital potenciometer wiper position (i.e., resistance lowers and more EMS is passing)
//           -> note that you have full EMS signal when the potentiometer wiper is at 0 (0 is the maximum)
//     "a": decreases EMS signal on channel 1 by increasing the digital potenciometer wiper position (i.e., resistance lowers and more EMS is passing)
//           -> note that you have no EMS signal when the potentiometer wiper is at 255 (255 is the maximum resistance of the potentiometer)
//     "w": increases EMS signal on channel 2 by decreasing the digital potenciometer wiper position (i.e., resistance lowers and more EMS is passing)
//           -> note that you have full EMS signal when the potentiometer wiper is at 0 (0 is the maximum)
//     "s": decreases EMS signal on channel 2 by increasing the digital potenciometer wiper position (i.e., resistance lowers and more EMS is passing)
//           -> note that you have no EMS signal when the potentiometer wiper is at 255 (255 is the maximum resistance of the potentiometer)
void doCommand(char c) {
      if (c == '1') {
		if (emsChannel1.isActivated()) {
			emsChannel1.deactivate();
      strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[1])));
			printer(buffer); //"\tEMS: Channel 1 inactive"
		} else {
			emsChannel1.activate();
      strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[0])));
			printer(buffer); //"\tEMS: Channel 1x active"
		}
	} else if (c == '2') {
		if (emsChannel2.isActivated()) {
			emsChannel2.deactivate();
      strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[3])));
			printer(buffer); //"\tEMS: Channel 2 inactive"
		} else {
			emsChannel2.activate();
      strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[2])));
			printer(buffer);  //"\tEMS: Channel 2 inactive"
		}
	} else if (c == 'a') {
		digitalPot.setPosition(1, digitalPot.getPosition(1) + 1);
    strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[4])));
		printer(
				buffer + String(digitalPot.getPosition(1))); //"\tEMS: Intensity Channel 1: "
	} else if (c == 'q') {
    strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[4])));
		digitalPot.setPosition(1, digitalPot.getPosition(1) - 1);
		printer(
				buffer + String(digitalPot.getPosition(1))); //"\tEMS: Intensity Channel 1: "
	} else if (c == 's') {
		//Note that this is channel 3 on Digipot but EMS channel 2
		digitalPot.setPosition(3, digitalPot.getPosition(3) + 1);
   strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[5])));
		printer(
				buffer + String(digitalPot.getPosition(3))); //"\tEMS: Intensity Channel 2: "
	} else if (c == 'w') {
		//Note that this is channel 3 on Digipot but EMS channel 2
		digitalPot.setPosition(3, digitalPot.getPosition(3) - 1);
   strcpy_P(buffer, (char*)pgm_read_word(&(string_table_outputs[5])));
		printer(
				buffer + String(digitalPot.getPosition(3))); //"\tEMS: Intensity Channel 2: "
	}
        else printer("\tERROR: SINGLE-CHAR Command Unknown");
}
