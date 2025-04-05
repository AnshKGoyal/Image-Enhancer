import streamlit as st

st.title("Multi-Stage Inference Pipeline Configuration")

# Upload the initial image
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

# Pipeline configuration section: allow the user to add multiple stages.
st.subheader("Define Your Pipeline Stages")

# Use session_state to hold list of stages: each stage is a dictionary with stage_order and model.
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []

# UI: Button to add a new stage.
if st.button("Add Stage"):
    st.session_state.pipeline.append({
        "stage": len(st.session_state.pipeline) + 1,
        "model": "Deblurring"  # Default value; you can later edit this.
    })
    
# Display the pipeline stages in order and allow user to change model selection.
for idx, stage in enumerate(st.session_state.pipeline):
    st.markdown(f"**Stage {stage['stage']}**")
    # Selectbox for model selection; in future, you could add more models.
    new_model = st.selectbox("Select Model", ["Deblurring", "Dehazing", "Other"],
                               index=["Deblurring", "Dehazing", "Other"].index(stage["model"]),
                               key=f"model_{idx}")
    st.session_state.pipeline[idx]["model"] = new_model

# Option to remove a stage if needed.
if st.button("Clear Pipeline"):
    st.session_state.pipeline = []

# Display final configuration for clarity.
st.write("Current Pipeline Configuration:", st.session_state.pipeline)

# When the user clicks "Run Pipeline", process through each stage sequentially.
if st.button("Run Pipeline"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not st.session_state.pipeline:
        st.error("Please add at least one stage to your pipeline.")
    else:
        # Initialize the processing pipeline:
        image_input = uploaded_file  # This will be passed to the first stage.
        outputs = []  # To store intermediate outputs.
        
        # For demonstration, we'll simulate calling a universal endpoint.
        # In a real scenario, you'd send a request to your FastAPI endpoint each time.
        for stage in st.session_state.pipeline:
            # For each stage, call the appropriate endpoint based on "model"
            # Here, you might choose the URL dynamically:
            if stage["model"] == "Deblurring":
                endpoint = "predict_deblur"
            elif stage["model"] == "Dehazing":
                endpoint = "predict_dehaze"
            else:
                endpoint = "predict_custom"  # For future models.
            
            # Simulate an API call or internal processing function
            # For instance, you might do:
            # response = requests.post(f"{BASE_URL}/{endpoint}/", files={"file": image_input}, data={"user_id": user_id}, headers=headers)
            # image_input = response.json().get("output_image")  # Next stage input.
            # outputs.append(response.json())
            
            # Here, just mimic the process:
            st.write(f"Processing stage {stage['stage']} using {stage['model']}...")
            # Simulated intermediate output: in a real case this would be a path or image bytes.
            intermediate_output = f"Output_{stage['model']}_{stage['stage']}"
            outputs.append(intermediate_output)
            # Use the simulated output as the next input (if needed)
            image_input = intermediate_output
            
        st.success("Pipeline complete!")
        st.write("Intermediate Outputs:", outputs)
        
        # Here you would display each intermediate image if available.
