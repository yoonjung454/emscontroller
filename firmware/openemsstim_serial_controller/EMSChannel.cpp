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
 * EMSChannel.cpp
 *
 *  Created on: 06.05.2014
 *  Author: Tim Duente
 *  Edit by: Max Pfeiffer 15.06.2015
 */
#include "EMSChannel.h"

//---------- constructor ----------------------------------------------------

EMSChannel::EMSChannel(uint8_t channel_to_Pads, uint8_t channel_to_Pads_2,
		uint8_t led_active_pin, AD5252* digitalPoti, uint8_t wiperIndex) {

	intensity = 0;
	state = OFF;

	maxIntensity = 215;
	minIntensity = 55;

	this->switch_1 = channel_to_Pads;
	this->switch_2 = channel_to_Pads_2;
	this->channel_active_led = led_active_pin;

	endTime = 0;
	deactivatingTime = 0;
	onTimeChannel = 1;

	this->digitalPoti = digitalPoti;
	this->wiperIndex = wiperIndex;

	pinMode(channel_to_Pads, OUTPUT);
	pinMode(channel_to_Pads_2, OUTPUT);
	pinMode(led_active_pin, OUTPUT);
	this->deactivateChannel();

}

EMSChannel::~EMSChannel() {

}

//---------- public ----------------------------------------------------
/*
 * Starts the communication with the digital Poti. Must be called for initialization.
 */
void EMSChannel::start() {
	Wire.begin();
}

/*
 * Routes the EMS signal to the electrodes
 */
void EMSChannel::activate() {
	digitalWrite(switch_1, HIGH);
	digitalWrite(switch_2, HIGH);
	state = ON;
	digitalWrite(channel_active_led, HIGH);
}

/*
 * Routes the EMS signal to the MOSFets
 */
void EMSChannel::deactivate() {
	if (state != DEACTIVATING) {
		digitalPoti->setPosition(wiperIndex, 255);
		state = DEACTIVATING;
		deactivatingTime = millis() + 50;
	}
}

/*
 * Routes the EMS signal to the MOSFets
 */
void EMSChannel::deactivateChannel() {
	digitalWrite(channel_active_led, LOW);
	digitalWrite(switch_2, LOW);
	digitalWrite(switch_1, LOW);
	endTime = 0;
	state = OFF;
}

/*
 * Proofs if the channel is active
 */
bool EMSChannel::isActivated() {
	return (state == ON);
}

/*
 * Sets the intensity from 0-100
 */
void EMSChannel::setIntensity(uint8_t intensity) {
	this->intensity = uint8_t(
			((maxIntensity - minIntensity) * intensity) * 0.01f + 0.5f)
			+ minIntensity;

	if (this->intensity > POTI_STEPS_UP) {
		this->intensity = POTI_STEPS_UP;
	} else if (this->intensity < POTI_STEPS_DOWN) {
		this->intensity = POTI_STEPS_DOWN;
	}

	uint8_t resistorLevel = POTI_STEPS_UP - this->intensity;

	digitalPoti->setPosition(wiperIndex, resistorLevel);
}

/*
 * Return the intensity in a range of 0-100
 */
int EMSChannel::getIntensity() {
	return intensity;
}

void EMSChannel::setSignalLength(int signalLength) {
	onTimeChannel = signalLength;

}

int EMSChannel::getSignalLength() {
	return onTimeChannel;

}

void EMSChannel::applySignal() {
	endTime = millis() + onTimeChannel;
}

int EMSChannel::check() {
	int channel_deactivated = 0;
	switch (state) {
	case ON:
		if (endTime && endTime <= millis()) {
			deactivate();
		}
		break;
	case DEACTIVATING:
		if (deactivatingTime && deactivatingTime <= millis()) {
			deactivateChannel();
			deactivatingTime = 0;

			channel_deactivated = 1;
		}
		break;
	case OFF:
		break;
	}

	return channel_deactivated;
}

//maxIntensity in percent
void EMSChannel::setMaxIntensity(int maxIntensity) {
	this->maxIntensity = uint8_t(POTI_STEPS_UP * maxIntensity * 0.01f + 0.5f);
}

//minIntensity in percent
void EMSChannel::setMinIntensity(int minIntensity) {
	this->minIntensity = uint8_t(POTI_STEPS_UP * minIntensity * 0.01f + 0.5f);
}

//---------- private ----------------------------------------------------
