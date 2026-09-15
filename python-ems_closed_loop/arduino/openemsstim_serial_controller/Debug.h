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
 * Debug.h
 *
 *  Created on: 21.02.2019
 *      Author: Tim Duente
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <Arduino.h>

//setup for verbose mode (prints debug messages if DEBUG_ON is 1). For a low latency mode, it should be set to 0.
#define DEBUG_ON 1

void debug_println(String msg);
void debug_print(String msg);

#endif /* DEBUG_H_ */
