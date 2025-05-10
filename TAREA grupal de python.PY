import cv2
import requests
import os
import time
from dotenv import load_dotenv

def main():
    # Cargar configuración de API desde archivo .env
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    API_ENDPOINT = os.getenv('API_ENDPOINT')
    
    if not API_KEY or not API_ENDPOINT:
        print("Error: Falta configurar API_KEY y API_ENDPOINT en el archivo .env")
        return
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    print("Presione ESPACIO para capturar imagen o ESC para salir")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara")
            break
        
        cv2.imshow('Detección de Personas - Presione ESPACIO para capturar', frame)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # Tecla ESC
            print("Programa finalizado por el usuario")
            break
        elif key == 32:  # Tecla ESPACIO
            # Guardar imagen temporal
            temp_image = 'temp_capture.jpg'
            cv2.imwrite(temp_image, frame)
            print("\nImagen capturada. Analizando...")
            
            # Enviar a la API de visión por computadora
            try:
                with open(temp_image, 'rb') as image_file:
                    image_data = image_file.read()
                
                headers = {
                    'Content-Type': 'application/octet-stream',
                    'Ocp-Apim-Subscription-Key': API_KEY
                }
                
                params = {
                    'visualFeatures': 'Objects,People',
                    'language': 'es'
                }
                
                response = requests.post(
                    f"{API_ENDPOINT}/vision/v3.1/analyze",
                    headers=headers,
                    params=params,
                    data=image_data
                )
                
                response.raise_for_status()
                analysis = response.json()
                
                # Procesar resultados
                people_count = len(analysis.get('people', []))
                objects = [obj['object'] for obj in analysis.get('objects', [])]
                
                print("\n=== RESULTADOS ===")
                print(f"Personas detectadas: {people_count}")
                if people_count > 0:
                    for i, person in enumerate(analysis['people'], 1):
                        print(f"  Persona {i}: Posición {person['rectangle']}")
                
                if objects:
                    print("\nObjetos detectados:")
                    print(", ".join(objects))
                else:
                    print("\nNo se detectaron objetos")
                
                # Mostrar imagen con resultados por 5 segundos
                if people_count > 0:
                    for person in analysis['people']:
                        x, y, w, h = person['rectangle']['x'], person['rectangle']['y'], \
                                    person['rectangle']['w'], person['rectangle']['h']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Persona', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9, (0, 255, 0), 2)
                
                cv2.imshow('Resultados', frame)
                cv2.waitKey(5000)  # Mostrar resultados por 5 segundos
                cv2.destroyWindow('Resultados')
                
            except Exception as e:
                print(f"Error al analizar la imagen: {str(e)}")
            
            # Eliminar imagen temporal
            try:
                os.remove(temp_image)
            except:
                pass
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()