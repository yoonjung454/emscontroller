/**
 * hardware_config.h
 * ------------------
 * This file does NOT invent any pin numbers. The actual pin wiring for the
 * EMS channels comes from the official openEMSstim firmware's own object
 * constructors, which already exist in openemsstim_serial_controller.ino:
 *
 *     EMSChannel emsChannel1(5, 4, A2, &digitalPot, 1);
 *     EMSChannel emsChannel2(6, 7, A3, &digitalPot, 3);
 *     AD5252     digitalPot(0);
 *
 * These four numbers (5, 4, A2 for channel 1; 6, 7, A3 for channel 2) and
 * the AD5252 I2C address offset (0) are the official openEMSstim example
 * wiring from https://github.com/PedroLopes/openEMSstim
 * (arduino-openEMSstim/arduino-openEMSstim.ino).
 *
 * >>> BEFORE YOU TRUST ANY OF THIS: open your own team's
 * >>> openemsstim_serial_controller.ino, find those same two `EMSChannel(...)`
 * >>> lines, and confirm the numbers there match how your board is actually
 * >>> soldered/wired. If your unit was wired differently, edit THOSE lines
 * >>> (not this file) to match your hardware. Nothing in this project will
 * >>> silently assume a pin mapping you haven't confirmed yourself.
 *
 * Everything in *this* file is our own add-on configuration -- channel
 * enable flags, protocol timing, and the Bluetooth on/off switch -- none of
 * which is hardware pin data.
 */

#ifndef HARDWARE_CONFIG_H_
#define HARDWARE_CONFIG_H_

// --- Channel availability -------------------------------------------------
// Reflects the team's current hardware status: channel 1 works, channel 2's
// pad/output is currently broken. Flip CHANNEL_2_ENABLED to true once it's
// repaired -- no other code changes are required for the serial protocol to
// start accepting channel-2 commands again.
#define CHANNEL_1_ENABLED true
#define CHANNEL_2_ENABLED false

// --- Bluetooth / RN4020 ----------------------------------------------------
// The prelim explicitly does not use BLE/RN4020. Set to 1 to restore the
// official firmware's Bluetooth init/read path unchanged if you revive it
// for the finals.
#define ENABLE_BLUETOOTH_MODULE 0

// --- Serial protocol -------------------------------------------------------
// Must match config.py's SerialConfig.baud_rate on the PC side.
#define SERIAL_BAUD_RATE 19200

// If no valid line arrives from the PC within this many ms while a channel
// is armed, the Arduino deactivates every channel on its own (independent of
// whatever the PC thinks is happening). This is the firmware-side half of
// the heartbeat safety requirement; serial_link.py / app.js send PING every
// ~400ms automatically, so 1500 is fine for real software.
//
// TEMPORARILY set to 10000 for manual Serial Monitor bench testing (a human
// typing commands by hand is much slower than 1500ms between lines). Set
// this back to 1500 before connecting the real PC software for actual runs
// -- a human typing is not an acceptable substitute for the real watchdog
// once anything is actually attached to a person.
#define HEARTBEAT_TIMEOUT_MS 1500

// Every SET command must carry a duration; this is the ceiling we enforce
// here even if a PC bug asks for longer. Mirrors the official firmware's own
// 5000ms cap inside EMSSystem::doActionCommand.
#define MAX_COMMAND_TTL_MS 5000

#endif /* HARDWARE_CONFIG_H_ */
