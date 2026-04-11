import time

try:
    import Jetson.GPIO as GPIO
    ON_JETSON = True
except ImportError:
    print("Jetson.GPIO nicht gefunden – Simulationsmodus.")
    ON_JETSON = False

# ── Pins (BCM-Schema) ────────────────────────────────────────────────────────
PIN_LED_R = 18   # Pin 18 → leuchtet blau  → rotes Bein hier einstecken
PIN_LED_G = 12   # Pin 12 → leuchtet grün  → stimmt bereits
PIN_LED_B = 15   # Pin 15 → leuchtet nichts → hier prüfenp

PINS = {
    'ROT   (Klasse C)': PIN_LED_R,
    'GRÜN  (Klasse A)': PIN_LED_G,
    'BLAU  (Klasse B)': PIN_LED_B,
}

def run_test():
    if ON_JETSON:
        GPIO.setmode(GPIO.BOARD)
        for pin in PINS.values():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    print("\nRGB-LED Test gestartet.")
    print("Jede Farbe leuchtet 2 Sekunden.\n")

    try:
        # Einmal alle drei Farben der Reihe nach durchschalten
        for color_name, pin in PINS.items():
            print(f"  → {color_name}")

            if ON_JETSON:
                GPIO.output(pin, GPIO.HIGH)

            time.sleep(2.0)  # 2 Sekunden leuchten lassen

            if ON_JETSON:
                GPIO.output(pin, GPIO.LOW)

            time.sleep(0.3)  # kurze Pause zwischen den Farben

        print("\nTest abgeschlossen. Alle LEDs aus.")

    finally:
        # GPIO immer sauber freigeben – auch wenn du Strg+C drückst
        if ON_JETSON:
            GPIO.cleanup()

if __name__ == '__main__':
    run_test()