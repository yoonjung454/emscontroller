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
 * EMSChannel.h
 *
 *  Created on: 06.05.2014
 *      Author: Tim Duente
 *  Edit by: Max Pfeiffer - 13.06.2015
 */

#ifndef EMSCHANNEL_H_
#define EMSCHANNEL_H_

#include <Arduino.h>
#include "Wire.h"
#include "AD5252.h"
#include "Debug.h"

// Individual for each device must be knew
//#define IDENT_PHRASE = "EMS-Service-BLE1"

//The Poti has 256 steps. 0 - 255.
#define POTI_STEPS_UP 255
#define POTI_STEPS_DOWN 0


class EMSChannel {

public:
	static void start();

	EMSChannel(uint8_t channel_to_Pads, uint8_t channel_to_Pads_2,
			uint8_t led_active_pin, AD5252 *digitalPoti, uint8_t wiperIndex);
	~EMSChannel();

	void activate();
	void deactivate();
	bool isActivated();

	void setIntensity(uint8_t intensity);
	int getIntensity();

	void setSignalLength(int signalLength);
	int getSignalLength();

	void applySignal();

	void setMaxIntensity(int maxIntensity);
	void setMinIntensity(int minIntensity);

	int check();

private:
	//internal state

	void deactivateChannel();
	uint8_t intensity;
	unsigned long int endTime;
	unsigned long int deactivatingTime;
	int onTimeChannel;

	uint8_t maxIntensity; // Poti Value
	uint8_t minIntensity; // Poti Value

	//Poti control variables
	AD5252* digitalPoti;
	uint8_t wiperIndex;

	//Pin connections to Solid State Relays and LED
	uint8_t switch_1;
	uint8_t switch_2;
	uint8_t channel_active_led;

	typedef enum {
		OFF, ON, DEACTIVATING
	}State;

	State state = OFF;


};

#endif /* EMSCHANNEL_H_ */
