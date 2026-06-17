#include <Wire.h> // бібліотека інтерфейсу I2C
#include <LiquidCrystal_I2C.h> // бібліотека дисплея для протоколу I2C
#include <ArduinoJson.h> // бібліотека для роботи з форматом JSON
#include <EEPROM.h> // бібліотека для роботи з енергонезалежною пам'яттю

// робочі піни датчиків летких органічних сполук MICS-5524
#define PIN1_MICS A1 // визначення піна для першого датчика
#define PIN2_MICS A2 // визначення піна для другого датчика
#define PIN3_MICS A3 // визначення піна для третього датчика
#define PIN4_MICS A0 // визначення піна для четвертого датчика

// Робочі піни анемометра і флюгера
#define PIN_ANEMOMETER A4 
#define PIN_VANE A5 
#define V_REF 5.0 // Напруга Arduino
#define OFFSET 0.14 // Шум анемометра

#define RESET_PIN 7 // пін для перезавантаження пристрою, з'єднаний з піном RESET

LiquidCrystal_I2C lcd(0x27, 20, 4); // об'єкт дисплею

// змінні для результатів розв'язання оберненої задачі (7 параметрів з інференсу)
float res_D = 0.0; // D0
float res_Q = 0.0; // Q0
float res_x = 0.0; // x0
float res_y = 0.0; // y0
float res_z = 0.0; // z0
float res_t = 0.0; // t0
float res_b = 0.0; // beta

String status_msg = "WAITING"; // статус виконання обчислень на сервері

unsigned long last_send_time = 0; // змінна-лічильник для таймера надсилання даних на сервер

struct __attribute__((packed)) DataPacket { // структура пакета даних для надсилання на сервер
  float c1;       // Концентрація 1
  float c2;       // Концентрація 2
  float c3;       // Концентрація 3
  float c4;       // Концентрація 4
  float wind_spd; // Швидкість вітру
  float wind_sin; // Напрямок вітру (синус кута з флюгера)
  float wind_cos; // Напрямок вітру (косинус кута з флюгера)
  float time_norm; // змінна-заглушка, нормований час проставляється на інференсі
};

struct __attribute__((packed)) ResponsePacket { // структура пакета даних відповіді від супутника
  uint8_t status_code; // статус виконаних обчислень на сервері
  float D0; 
  float Q0; 
  float x0; 
  float y0; 
  float z0; 
  float t0; 
  float beta; 
};

const int EEPROM_ADDR = 0; // змінна-адреса в енергонезалежній пам'яті
bool DEBUG_MODE = false; // директива для переведення пристрою в режим налагодження по WIFI

// змінні для режиму WIFI
#include "WiFiS3.h" // бібліотека WIFI-модуля ESP32S3
char ssid[] = "Shnur"; // SSID мережі
char pass[] = "shnur01092004"; // пароль для мережі
IPAddress server(78, 17, 34, 80); // адреса обчислювального сервера
int port = 8000; // порт, на якому знаходиться API
WiFiClient client; // об'єкт WIFI-клієнта

// змінні для режиму супутника
#include <IridiumSBD.h> // бібліотека супутникового модему Iridium RockBLOCK 9602
#define IridiumSerial Serial1 // додатковий фізичний послідовний порт плати Arduino UNO R4 WIFI
IridiumSBD satmodem(IridiumSerial); // об'єкт супутникового модему Iridium RockBLOCK 9602
int modemHardwareErr; // змінна апаратної помилки модема Iridium RockBLOCK 9602

unsigned long send_interval; // інтервал надсилання пакетів на сервер (ініціалізується залежно від режиму)

// функція виведення інформації про критичну помилку або службове повідомлення та перезавантаження пристрою
void systemReset(String msg, bool isservice=true) { 
  // виведення даних в послідовний порт
  if (!isservice){
    Serial.print("FATAL ERROR: ");
  }
  Serial.println(msg);
  Serial.println("RESTARTING...");
  // виведення даних на дисплей 
  lcd.clear();
  lcd.setCursor(0, 0);
  if (!isservice){
    lcd.print("FATAL ERROR!");
  }
  lcd.setCursor(0, 1);
  lcd.print(msg);
  lcd.setCursor(0, 2);
  lcd.print("RESTARTING...");
  delay(5000);
  pinMode(RESET_PIN, OUTPUT); // подаємо GND на пін RESET
  delay(250);
}

// ініціалізація компонентів
void setup() {
  byte modeFromEEPROM = EEPROM.read(EEPROM_ADDR); // ініціалізація порожньої енергонезалежної пам'яті значенням за замовчуванням
  if (modeFromEEPROM == 255) {
    modeFromEEPROM = 0;
    EEPROM.write(EEPROM_ADDR, 0);
  }
  
  //DEBUG_MODE = (modeFromEEPROM == 1); // встановлюємо режим в залежності від значення в EEPROM
  DEBUG_MODE = 1; // для демонстрації хардкодимо в DEBUG
  send_interval = 10000; // інтервал для посилання даних на інференс
  
  Serial.begin(9600); // ініціалізація основного послідовного порту
  lcd.init(); // ініціалізація дисплею
  lcd.backlight(); // увімкнення підсвітки дисплею

  // Виведення вітального екрана
  lcd.setCursor(4, 1);
  lcd.print("InvProbMeter");
  lcd.setCursor(5, 2);
  lcd.print("NUWEE 2026");
  delay(3000);
  lcd.clear();

  if (DEBUG_MODE){ // підключення до заданої WIFI-мережі
    Serial.println("MODE: WIFI DEBUG");
    lcd.setCursor(0, 0);
    lcd.print("MODE: WIFI DEBUG");

    if (WiFi.status() == WL_NO_MODULE){ // перевірка наявності апаратного модуля WIFI
      systemReset("HARDWARE ERROR"); // перезавантаження, якщо модуль відсутній
    }

    Serial.println("CONNECTING..."); // якщо апаратний модуль є - йдемо далі
    lcd.setCursor(0, 1);
    lcd.print("CONNECTING...");

    int attempts = 0; // змінна-лічильник спроб підключення

    while (WiFi.status() != WL_CONNECTED) { // цикл поки немає з'єднання
      WiFi.begin(ssid, pass); // спроба підключення до заданої мережі WIFI
      Serial.print("."); // індикація поточної спроби
      delay(5000);
      if (attempts++ > 5){
        systemReset("NETWORK ERROR"); // перезавантаження, якщо після 5 спроб підключення не вдалось
      }
    }

    Serial.println("\nCONNECTED!"); // якщо пройдено усі перевірки - вивід повідомлення про успішне підключення
    lcd.setCursor(0, 1);
    lcd.print("CONNECTED!        ");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    lcd.setCursor(0, 2);
    lcd.print("IP: ");
    lcd.print(WiFi.localIP());

    delay(2000); // затримка для стабілізації
    lcd.clear();

  } else { // підключення до супутника Iridium
    Serial.println("MODE: IRIDIUM");
    lcd.setCursor(0, 0);
    lcd.print("MODE: IRIDIUM");

    IridiumSerial.begin(19200); // ініціалізація фізичного послідовного порту для модему

    Serial.println("CONNECTING...");
    lcd.setCursor(0, 1);
    lcd.print("CONNECTING...");
    
    modemHardwareErr = satmodem.begin(); // ініціалізація модему
    if (modemHardwareErr != ISBD_SUCCESS) { // перевірка успішності ініціалізації
      systemReset("HARDWARE ERROR " + String(modemHardwareErr)); // перезавантаження, якщо виникла апаратна проблема
    }

    int signalQuality = -1; // змінна для якості сигналу
    int modemQualityErr = satmodem.getSignalQuality(signalQuality); // отримання якості сигналу

    Serial.print("SIGNAL: ");
    Serial.println(signalQuality);
    lcd.setCursor(0, 2);
    lcd.print("SIGNAL: ");
    lcd.print(signalQuality);

    Serial.println("CONNECTED!"); // якщо пройдено усі перевірки - вивід повідомлення про успішне підключення
    lcd.setCursor(0, 1);
    lcd.print("CONNECTED!        ");

    delay(2000);
    lcd.clear();
  }
}

// основний робочий цикл
void loop() {

  // прийом команд від користувача через послідовний порт:
  if (Serial.available() > 1){
    char incoming = Serial.read();
    int value = Serial.parseInt();
    switch (incoming){
      case 'd': {
        EEPROM.write(EEPROM_ADDR, 1);
        systemReset("ENTERING DEBUG MODE", true);
        break;
      }
      case 'i': {
        EEPROM.write(EEPROM_ADDR, 0);
        systemReset("LEAVING DEBUG MODE", true);
        break;
      }
    }
  }

  // зняття концентрацій CO2 з кожного датчика MICS-5524 в сирих одиницях АЦП
  float ppm1 = analogRead(PIN1_MICS);
  float ppm2 = analogRead(PIN2_MICS);
  float ppm3 = analogRead(PIN3_MICS);
  float ppm4 = analogRead(PIN4_MICS);

  // Анемометр
  // int raw_speed = analogRead(PIN_ANEMOMETER); закомент до покупки мультиплексора
  int raw_speed = 0;
  float voltage_speed = (raw_speed / 1023.0) * V_REF;
  float wind_ms = (voltage_speed - OFFSET) * 12.0;
  if (wind_ms < 0) wind_ms = 0; // Фільтр від'ємних значень
    
  // Флюгер
  // int raw_dir = analogRead(PIN_VANE); закомент до покупки мультиплексора
  int raw_dir = 0;
  float voltage_dir = (raw_dir / 1023.0) * V_REF;
  float wind_degrees = (voltage_dir / V_REF) * 6.28318530718;

  // обмін пакетами через період send_interval
  if (millis() - last_send_time >= send_interval) {
    last_send_time = millis();

    // виведення значень концентрації та вітру у послідовний порт
    Serial.print(" C1: "); Serial.print(ppm1);
    Serial.print(" C2: "); Serial.print(ppm2);
    Serial.print(" C3: "); Serial.print(ppm3);
    Serial.print(" C4: "); Serial.println(ppm4);
    Serial.print(" WIND SPD: "); Serial.print(wind_ms);
    Serial.print(" WIND DIR: "); Serial.print(wind_degrees);
    Serial.print(" SIN(WIND DIR): "); Serial.print(sin(wind_degrees));
    Serial.print(" COS(WIND DIR): "); Serial.println(cos(wind_degrees));

    // виведення концентрації та результатів розв'язання оберненої задачі на дисплей
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("X:");
    lcd.print(res_x, 1);
    lcd.print(" Y:");
    lcd.print(res_y, 1);
    lcd.setCursor(0, 1);
    lcd.print("Z:");
    lcd.print(res_z, 1);
    lcd.print(" Q0:");
    lcd.print(res_Q, 1); 
    lcd.setCursor(0, 2);
    lcd.print("D0:");
    lcd.print(res_D, 1);
    lcd.print(" b:");
    lcd.print(res_b, 1);
    lcd.setCursor(0, 3);
    lcd.print("t:");
    lcd.print(res_t, 1);
    lcd.print(" " );
    lcd.print(status_msg);
    
    DataPacket payload; // пакет даних для відправки на сервер
    payload.c1 = (float)ppm1; // концентрація з першого MQ135
    payload.c2 = (float)ppm2; // концентрація з другого MQ135
    payload.c3 = (float)ppm3; // концентрація з третього MQ135
    payload.c4 = (float)ppm4; // концентрація з четвертого MQ135
    payload.wind_spd = wind_ms; // швидкість вітру
    payload.wind_sin = sin(wind_degrees); // напрямок (синус)
    payload.wind_cos = cos(wind_degrees); // напрямок (косинус)
    payload.time_norm = 0.0; //заглушка

    if(DEBUG_MODE){ // вибір методу відправки
      sendViaWiFiHex(payload); // відправка через WIFI
    } else {
      sendViaIridium(payload); // відправка через супутник
    }
  }
}

void sendViaWiFiHex(DataPacket pkt) { // функція обміну даними через WIFI
  if (WiFi.status() != WL_CONNECTED){ // перевірка з'єднання
    return; // вихід, якщо немає з'єднання
  }

  if (client.connect(server, port)) { // спроба підключення до сервера
    uint8_t *raw = (uint8_t*)&pkt; // вказівник на дані пакета
    String hexStr = ""; // рядок для HEX даних

    for(int i=0; i<sizeof(pkt); i++) { // рухаємось побайтово у пакеті даних
      if(raw[i] < 16) { // якщо число менше 16
        hexStr += "0"; // додаємо ведучий нуль
      }
      hexStr += String(raw[i], HEX); // додаємо HEX значення байта
    }

    String jsonBody = "{\"data\":\"" + hexStr + "\"}"; // формування результуючого JSON
    Serial.print("[HTTP OUT] ");
    Serial.println(jsonBody); // виведення JSON у послідовний порт

    client.println("POST /predict HTTP/1.1"); // API-endpoint та метод запиту
    client.print("Host: "); // заголовок хоста
    client.println(server); // адреса сервера
    client.println("Content-Type: application/json"); // тип контенту
    client.print("Content-Length: "); // довжина контенту
    client.println(jsonBody.length()); // значення довжини
    client.println("Connection: close"); // закриття з'єднання
    client.println(); // кінець заголовків
    client.println(jsonBody); // тіло запиту

    unsigned long timeout = millis(); // змінна-лічильник для очікування відповіді від сервера
    while (client.connected() && millis() - timeout < 2000) {
      if (client.available()) { // якщо є залишкові дані
        String line = client.readStringUntil('\n'); // читаємо їх в тимчасову рядкову змінну
        if (line == "\r"){ // якщо кінець заголовків - вихід
          break;
        }
      }
    }
    
    String response = client.readString(); // читання відповіді сервера
    
    Serial.print("[HTTP IN] "); // виведення відповіді сервера у послідовний порт
    Serial.println(response);

    StaticJsonDocument<512> doc; // буфер JSON
    DeserializationError error = deserializeJson(doc, response); // парсинг JSON

    if (!error) { // якщо немає помилок парсингу
      const char* status = doc["status"]; // отримання статусу проведених обчислень
      status_msg = String(status); // оновлення статусу на дисплеї
      
      if (strcmp(status, "SOLVED") == 0) { // якщо розрахунок успішний
        res_D = doc["result"]["D0"]; // запис результату D0
        res_Q = doc["result"]["Q0"]; // запис результату Q0
        res_x = doc["result"]["x0"]; // запис результату x0
        res_y = doc["result"]["y0"]; // запис результату y0
        res_z = doc["result"]["z0"]; // запис результату z0
        res_t = doc["result"]["t0"]; // запис результату t0
        res_b = doc["result"]["beta"]; // запис результату beta
      }
    }
    client.stop(); // відключення
  }
}

void sendViaIridium(DataPacket pkt) { // функція обміну даними через Iridium
  status_msg = "SENDING..."; // оновлення статусу

  // виведення пакета у послідовний порт
  Serial.print("[SATELITE TX] SENDING: ");
  Serial.print("C1:"); Serial.print(pkt.c1);
  Serial.print(" | C2:"); Serial.print(pkt.c2);
  Serial.print(" | C3:"); Serial.print(pkt.c3);
  Serial.print(" | C4:"); Serial.println(pkt.c4);
  Serial.print("WIND SPEED:"); Serial.print(pkt.wind_spd);
  Serial.print(" | SIN(WIND DIRECTION):"); Serial.print(pkt.wind_sin);
  Serial.print(" | COS(WIND DIRECTION):"); Serial.println(pkt.wind_cos);

  uint8_t rxBuffer[64]; // буфер прийому
  size_t rxSize = sizeof(rxBuffer); // розмір буфера
  
  int err = satmodem.sendReceiveSBDBinary((uint8_t*)&pkt, sizeof(pkt), rxBuffer, rxSize); // обмін даними із супутником
  
  if (err == ISBD_SUCCESS) { // якщо успішно
    Serial.print("[SATTELITE RX] Bytes received: ");
    Serial.println(rxSize); // кількість байтів
    if (rxSize > 0) { // якщо є дані
      if (rxSize == sizeof(ResponsePacket)) { // якщо розмір правильний
        ResponsePacket* resp = (ResponsePacket*)rxBuffer; // перетворення буфера у тип sturct
        
        res_D = resp->D0; // оновлення D0
        res_Q = resp->Q0; // оновлення Q0
        res_x = resp->x0; // оновлення x0
        res_y = resp->y0; // оновлення y0
        res_z = resp->z0; // оновлення z0
        res_t = resp->t0; // оновлення t0
        res_b = resp->beta; // оновлення beta

        // виведення відповіді від сервера у послідовний порт
        Serial.print("[DATA] Status Code: ");
        Serial.println(resp->status_code);
        Serial.print("[DATA] D0="); Serial.print(res_D, 3);
        Serial.print(" | Q0="); Serial.print(res_Q, 0);
        Serial.print(" | Beta="); Serial.println(res_b, 2);

        if (resp->status_code == 1){ // якщо код 1
          status_msg = "SOLVED"; // статус вирішено
        }
        else if (resp->status_code == 0){ // якщо код 0
          status_msg = "WAITING"; // статус очікування
        }
        else if (resp->status_code == 2){ // якщо код 2
          status_msg = "CLIENT ERROR"; // помилка клієнта
        }
        else if (resp->status_code == 3){ // якщо код 3
          status_msg = "SERVER ERROR"; // помилка сервера
        }
      } else { // якщо розмір неправильний
        status_msg = "BAD SIZE"; // помилка розміру
        Serial.print("[SATTELITE RX] BAD SIZE: Expected ");
        Serial.print(sizeof(ResponsePacket));
        Serial.print(" BYTES, GOT ");
        Serial.println(rxSize);
      }
    } else { // якщо буфер порожній
      status_msg = "NO RX"; // немає відповіді
      Serial.println("[SATTELITE RX] DATA SENT, BUT RX BUFFER IS EMPTY (NO REPLY)");
    }
  } else { // якщо помилка модема
    status_msg = "NET ERROR: " + String(err);
    Serial.print("MODEM ERROR: ");
    Serial.println(err);
  }
}
