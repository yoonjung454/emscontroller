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
 * Rn4020BTLe.h
 *
 *  Created on: 19.03.2015
 *      Author: Tim Duente
 */

#include "AltSoftSerial.h"
#include "Debug.h"

#ifndef RN4020_RN4020BTLE_H_
#define RN4020_RN4020BTLE_H_

#include <Arduino.h>

class Rn4020BTLe {
public:
	Rn4020BTLe(uint8_t HW_Wake_Up, AltSoftSerial *serial);
	virtual ~Rn4020BTLe();
	void reset();
	void init(String bluetoothName);
	void start_communication();

private:
	bool print_response();
  bool check_baudrate(unsigned long baudrate);

	uint8_t HW_Wake_Up;
	AltSoftSerial *serial;


};

#endif /* RN4020_RN4020BTLE_H_ */
