/**
 * hardware_config.h
 * ------------------
 * This file does NOT invent any pin numbers. The actual pin wiring for the
 * EMS channels comes from openemsstim_serial_controller.ino's own object
 * constructors:
 *
 *     EMSChannel emsChannel1(6, 7, A3, &digitalPot, 3);
 *     EMSChannel emsChannel2(5, 4, A2, &digitalPot, 1);
 *     AD5252     digitalPot(0);
 *
 * NOTE: this is the SWAPPED order relative to the original openEMSstim
 * example (which pairs (5,4,A2)=channel1 / (6,7,A3)=channel2). We confirmed
 * with LED/multimeter + on-arm testing that on our actual board, the output
 * physically labeled "channel 1" comes out of the (6,7,A3) wiring, not
 * (5,4,A2) -- so the two EMSChannel(...) lines were swapped to match.
 *
 * >>> If you rewire the board or swap to a different unit, re-verify this
 * >>> with LED/multimeter before trusting it again -- don't assume the
 * >>> mapping above still holds.
 *
 * Everything in *this* file is our own add-on configuration -- channel
 * enable flags, protocol timing, and the Bluetooth on/off switch -- none of
 * which is hardware pin data.
 */

#ifndef HARDWARE_CONFIG_H_
#define HARDWARE_CONFIG_H_

// --- Channel availability -------------------------------------------------
// Channel 2's hardware was repaired and re-tested -- both channels enabled.
// If channel 2 breaks again, flip this back to false; no other code changes
// are needed for the serial protocol to stop/start accepting channel-2
// commands.
#define CHANNEL_1_ENABLED true
#define CHANNEL_2_ENABLED true

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
