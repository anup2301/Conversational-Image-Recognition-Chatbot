import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyCRbn2FDmNLcWVq22lQKe_3ibdRcy-w1KU")

yolo_model = YOLO('yolov8s.pt')

def detect_objects(image):
    results = yolo_model(image)
    detections = results[0].boxes
    img_with_boxes = np.array(image)
    detected_objects = []

    for detection in detections:
        x_min, y_min, x_max, y_max = detection.xyxy.numpy().astype(int).flatten()
        class_id = int(detection.cls[0])
        label = yolo_model.names[class_id]
        confidence = detection.conf[0].item()
        color = (0, 255, 0)
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img_with_boxes, f"{label} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detected_objects.append(label)
    
    return img_with_boxes, detected_objects

def chatbot_response(prompt):
    try:
        response = genai.generate_text(prompt=prompt)
        if response.candidates:
            return response.candidates[0]['output']
        return "No text generated. Please try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Tech Bandits Object Detection Tool with Chatbot")
    st.write("Upload an image to perform object detection using YOLOv8.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_with_boxes, detected_objects = detect_objects(img)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.image(img_with_boxes, caption='Detected Objects', use_column_width=True)

        if detected_objects:
            st.write("### Object Descriptions")
            objects_description = chatbot_response(f"Can you give a short description of the following objects: {', '.join(detected_objects)}?")
            st.write(objects_description)

    st.write("## Chat with Bandits chatbot")
    user_input = st.text_input("Ask Gemini AI anything:")

    if st.button("Send"):
        if user_input:
            conversation_response = chatbot_response(user_input)
            st.write("### Chatbot Response")
            st.write(conversation_response)
        else:
            st.write("Please enter a question or prompt for the chatbot.")

if __name__ == "__main__":
    main()
