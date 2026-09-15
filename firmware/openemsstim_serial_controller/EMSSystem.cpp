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

/*
 * EMSSystem.cpp
 *
 *  Created on: 26.05.2014
 *  Author: Tim Duente
 *  Edit by: Max Pfeiffer - 13.06.2015
 */

#include "EMSSystem.h"

EMSSystem::EMSSystem(uint8_t channels) {
	emsChannels = (EMSChannel**) malloc(channels * sizeof(EMSChannel*));
	maximum_channel_count = channels;
	current_channel_count = 0;
}

EMSSystem::~EMSSystem() {
	free(emsChannels);
}

void EMSSystem::addChannelToSystem(EMSChannel *emsChannel) {
	if (current_channel_count < maximum_channel_count) {
		emsChannels[current_channel_count] = emsChannel;
		current_channel_count++;
	}
}

// get the next number out of a String object and return it
int EMSSystem::getNextNumberOfString(String *command, uint8_t startIndex) {
	int value = 0;
	bool valid = false;
	// Select the number in the string
	for (uint8_t i = startIndex + 1; i < command->length(); i++) {
		char tmp = command->charAt(i);
		if (tmp >= '0' && tmp <= '9') {
			value = value * 10 + (tmp - '0');
			valid = true;
		} else {
			break;
		}
	}
	if (valid)
		return value;
	else
		return -1;
}

void EMSSystem::doActionCommand(String *command) {
	int seperatorChannel = -1;
	int seperatorSignalLength = -1;
	int seperatorSignalIntensity = -1;

	if (command->length() > 0) {

		// Channel
		seperatorChannel = command->indexOf(CHANNEL);
		int currentChannel = -1;

		if (seperatorChannel != -1) {

			currentChannel = getNextNumberOfString(command, seperatorChannel);
			//Serial.print("Channel");
			//Serial.println(currentChannel);
		}

		// Signal length onTime
		seperatorSignalLength = command->indexOf(TIME);
		int signalLength = -1;
		if (seperatorSignalLength != -1) {
			signalLength = getNextNumberOfString(command,
					seperatorSignalLength);
			if (signalLength > 5000) {
				//signaleLength max 5000ms
				signalLength = 5000;
			}
			emsChannels[currentChannel]->setSignalLength(signalLength);
		}

		// Signal Intensity
		seperatorSignalIntensity = command->indexOf(INTENSITY);
		int signalIntensity = -1;
		if (seperatorSignalIntensity != -1) {
			signalIntensity = getNextNumberOfString(command,
					seperatorSignalIntensity);
			emsChannels[currentChannel]->setIntensity(signalIntensity - 1);
		}

		// Apply the command
		//int seperatAction = command->indexOf(ACTION);
		//bool action = false;
//		if (seperatAction != -1) {
//			action = true;
//		}

		if (currentChannel >= 0 && currentChannel < current_channel_count) {
			emsChannels[currentChannel]->activate();
			emsChannels[currentChannel]->applySignal();
		} else {
			//deactivate all channels if channelNumber is wrong
			shutDown();
		}

	} else {
		debug_println(F("Command == NULL!"));
	}

}

void EMSSystem::shutDown() {
	for (int i = 0; i < current_channel_count; i++) {
		emsChannels[i]->deactivate();
	}
}

/* TODO change to set commands */

void EMSSystem::setOption(String *option) {
	char secChar = option->charAt(2);
	int channel = -1;
	int value = -1;
	switch (option->charAt(1)) {
	case 'C':
		if (secChar == 'T' && getChannelAndValue(option, &channel, &value)) {
			//set changeTime
			//emsChannels[channel]->setIncreaseDecreaseTime(value);
		}
		break;
	case 'M':
		if (secChar == 'A' && getChannelAndValue(option, &channel, &value)) {
			//Maxixum value for the calibration
			emsChannels[channel]->setMaxIntensity(value);
		} else if (secChar == 'I'
				&& getChannelAndValue(option, &channel, &value)) {
			//Minimum value for the calibration
			emsChannels[channel]->setMinIntensity(value);
		}
		break;

	default:
		break;
	}

}

bool EMSSystem::getChannelAndValue(String *option, int *channel, int *value) {
	int left = option->indexOf('[');
	int right = option->lastIndexOf(']');
	int seperator = option->indexOf(',', left + 1);

	if (left < seperator && seperator < right && left != -1 && right != -1
			&& seperator != -1) {
		String help = option->substring(left + 1, seperator);
		(*channel) = help.toInt();
		help = option->substring(seperator + 1, right);
		(*value) = help.toInt();

		//Parsing successful
		//Check whether channel exists
		return isInRange((*channel));
	}
//Parsing not successful
	return false;
}

bool EMSSystem::isInRange(int channel) {
	return (channel >= 0 && channel < current_channel_count);
}

uint8_t EMSSystem::check() {
	uint8_t stopCount = 0;
	for (uint8_t i = 0; i < current_channel_count; i++) {
		stopCount = stopCount + emsChannels[i]->check();
	}
	return stopCount;
}

void EMSSystem::doCommand(String *command) {
	if (command->length() > 0) {
		if (command->indexOf(ACTION) != -1) {
			doActionCommand(command);
		} else if (command->charAt(0) == OPTION) {
			setOption(command);
		} else {
			debug_print("EMS SYSTEM: Unknown command: ");
			debug_println((*command));
		}
	}
}

void EMSSystem::start() {
	EMSChannel::start();
}
