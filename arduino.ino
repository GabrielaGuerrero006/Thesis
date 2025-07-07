const int pin7 = 7;
const int pin12 = 12;
const int pin13 = 13;

void setup() {
  Serial.begin(9600);
  pinMode(pin7, OUTPUT);
  pinMode(pin12, OUTPUT);
  pinMode(pin13, OUTPUT);

  // Inicializa los pines en estado LOW
  digitalWrite(pin7, LOW);
  digitalWrite(pin12, LOW);
  digitalWrite(pin13, LOW);

  Serial.println("Arduino listo.");
}

void loop() {
  if (Serial.available() > 0) {
    // Lee la cadena hasta el carácter de nueva línea
    String command = Serial.readStringUntil('\n'); 
    command.trim(); // Elimina cualquier espacio en blanco (incluido el '\r' si viene de Windows)

    Serial.print("Comando recibido: ");
    Serial.println(command);

    if (command.length() >= 2) { // Espera formatos como "7H", "12L", "13H"
      int pinNum;
      char stateChar = command.charAt(command.length() - 1); // Último carácter es el estado (H/L)
      String pinStr = command.substring(0, command.length() - 1); // El resto es el número de pin

      pinNum = pinStr.toInt(); // Convierte la subcadena del pin a entero

      int state = LOW;
      if (stateChar == 'H') {
        state = HIGH;
      } else if (stateChar == 'L') {
        state = LOW;
      } else {
        Serial.println("Caracter de estado inválido.");
        return;
      }

      switch (pinNum) {
        case 7:
          digitalWrite(pin7, state);
          Serial.print("Pin 7 establecido a ");
          Serial.println(state == HIGH ? "HIGH" : "LOW");
          break;
        case 12:
          digitalWrite(pin12, state);
          Serial.print("Pin 12 establecido a ");
          Serial.println(state == HIGH ? "HIGH" : "LOW");
          break;
        case 13:
          digitalWrite(pin13, state);
          Serial.print("Pin 13 establecido a ");
          Serial.println(state == HIGH ? "HIGH" : "LOW");
          break;
        default:
          Serial.print("Número de pin inválido: ");
          Serial.println(pinNum);
          break;
      }
    } else {
      Serial.println("Formato de comando inválido.");
    }
  }
}