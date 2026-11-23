
#include <Arduino.h>

#define ENCODER_IZQ_A 34
#define ENCODER_DER_A 32

volatile long pulsosIzq = 0;
volatile long pulsosDer = 0;

void IRAM_ATTR contarIzq() {
    pulsosIzq++;
}

void IRAM_ATTR contarDer() {
    pulsosDer++;
}

void setup() {
    Serial.begin(115200);

    pinMode(ENCODER_IZQ_A, INPUT);
    pinMode(ENCODER_DER_A, INPUT);

    attachInterrupt(
        digitalPinToInterrupt(ENCODER_IZQ_A),
        contarIzq,
        RISING
    );

    attachInterrupt(
        digitalPinToInterrupt(ENCODER_DER_A),
        contarDer,
        RISING
    );

    Serial.println("PRUEBA DE ENCODERS");
    Serial.println("Gira las ruedas manualmente.");
    Serial.println("Escribe R para reiniciar el conteo.");
}

void loop() {
    static unsigned long anterior = 0;

    if (millis() - anterior >= 500) {
        anterior = millis();

        noInterrupts();
        long izq = pulsosIzq;
        long der = pulsosDer;
        interrupts();

        Serial.print("Izquierdo: ");
        Serial.print(izq);

        Serial.print(" | Derecho: ");
        Serial.println(der);
    }

    if (Serial.available()) {
        char tecla = Serial.read();

        if (tecla == 'R' || tecla == 'r') {
            noInterrupts();
            pulsosIzq = 0;
            pulsosDer = 0;
            interrupts();

            Serial.println("Conteo reiniciado.");
        }
    }
}
