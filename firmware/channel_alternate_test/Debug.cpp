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
 * Debug.c
 *
 *  Created on: 21.02.2019
 *      Author: Tim Duente
 */

#include "Debug.h"

/*
 * helper println function that handles the DEBUG_ON flag automatically
 */
void debug_println(String msg) {
	if (DEBUG_ON) {
		Serial.println(msg);
		Serial.flush();
	}
}

/*
 * helper print function that handles the DEBUG_ON flag automatically
 */
void debug_print(String msg) {
	if (DEBUG_ON) {
		Serial.print(msg);
		Serial.flush();
	}
}
