import streamlit as st
import requests
import json
from PIL import Image

# Constants for API endpoints
BASE_URL = "http://127.0.0.1:8000"
LOGIN_URL = f"{BASE_URL}/login"
REGISTER_URL = f"{BASE_URL}/users/"
PREDICT_PIPELINE_URL = f"{BASE_URL}/predict_pipeline/"
GLOBAL_OUTPUTS_URL = f"{BASE_URL}/model_outputs/global/"

# Helper function to persist token in URL query parameters
def persist_token_in_url(token: str):
    st.query_params.token = token

# Helper function to get token from URL query parameters
def get_token_from_url():
    params = st.query_params
    if "token" in params:
        return params.get("token")
    return None

# Helper function to perform login
def login(email: str, password: str):
    data = {
        "username": email,
        "password": password
    }
    response = requests.post(LOGIN_URL, data=data)
    if response.status_code == 200:
        st.session_state["user_id"] = response.json().get("user_id")
        return response.json().get("access_token")
    else:
        st.error("Login failed. Please check your email and password.")
        return None

# Helper function to register a new user
def register(username: str, email: str, password: str):
    data = {
        "username": username,
        "email": email,
        "password": password
    }
    response = requests.post(REGISTER_URL, json=data)
    if response.status_code == 200:
        return True
    else:
        st.error(f"Registration failed: {response.text}")
        return False

# Helper function to make authenticated API calls
def api_post(url, data=None, files=None):
    response = requests.post(url, data=data, files=files,headers={"token" : st.session_state["access_token"]})
    return response

def api_get(url):
    response = requests.get(url,headers={"token" : st.session_state["access_token"]})
    return response

def show_login_page():
    st.title("Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if not email:
            st.error("Email cannot be empty.")
        elif not password:
            st.error("Password cannot be empty.")
        else:
            token = login(email, password)
            if token:
                st.session_state["access_token"] = token
                persist_token_in_url(token)
                st.success("Logged in successfully!")
                st.rerun()  

def show_register_page():
    st.title("Register")
    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    if st.button("Register", key="register_button"):
        if not username:
            st.error("Username cannot be empty.")
        elif not email:
            st.error("Email cannot be empty.")
        elif not password:
            st.error("Password cannot be empty.")
        else:
            success = register(username, email, password)
            if success:
                st.success("Registered successfully! Please switch to the Login tab to log in.")

def show_main_app():
    st.title("Multi-Stage Image Processing Pipeline")
    # Sidebar menu (excluding Logout)
    menu = st.sidebar.radio("Menu", ["Configure Pipeline", "View Global Outputs", "View User Outputs"])

    if menu == "Configure Pipeline":
        configure_pipeline_page()
    elif menu == "View Global Outputs":
        global_outputs_page()
    elif menu == "View User Outputs":
        user_outputs_page()


    # Logout button (separate from menu)
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

# Logout function
def user_outputs_page():
    st.header("My Model Outputs")

    user_outputs_url = f"{BASE_URL}/model_outputs/user/" # User-specific endpoint
    response = api_get(user_outputs_url)
    if response.status_code != 200:
        st.error(f"Failed to retrieve user outputs: {response.text}")
        return

    outputs = response.json()
    display_outputs(outputs)


# Logout function
def logout():
    if "access_token" in st.session_state:
        del st.session_state["access_token"]
    del st.query_params.token  # Clear query params
    st.sidebar.success("Logged out successfully.")

def configure_pipeline_page():
    st.header("Configure Your Pipeline")

    # Upload the initial image
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Please upload an image to proceed.")
        return

    # Initialize pipeline stages
    if "pipeline_stages" not in st.session_state:
        st.session_state.pipeline_stages = []

    # Add new stage
    if st.button("Add Stage"):
        st.session_state.pipeline_stages.append({
            "stage": len(st.session_state.pipeline_stages) + 1,
            "model": "deblur"
        })

    # Display and configure pipeline stages
    if st.session_state.pipeline_stages:
        for idx, stage in enumerate(st.session_state.pipeline_stages):
            st.subheader(f"Stage {stage['stage']}")
            model_choice = st.selectbox(
                "Select Model",
                ["deblur", "dehaze"],
                key=f"model_choice_{idx}"
            )
            st.session_state.pipeline_stages[idx]["model"] = model_choice

        # Option to clear the pipeline
        if st.button("Clear Pipeline"):
            st.session_state.pipeline_stages = []

        # Run Pipeline
        if st.button("Run Pipeline"):
            run_pipeline(uploaded_file, st.session_state.pipeline_stages)

def run_pipeline(uploaded_file, pipeline_stages):
    st.info("Processing your pipeline. Please wait...")
    pipeline_config = pipeline_stages
    pipeline_json = json.dumps(pipeline_config)

    data = {
        "user_id": int(st.session_state["user_id"]),
        "pipeline": pipeline_json
        
    }
    files = {"file": uploaded_file} # Send UploadFile object directly

    response = api_post(PREDICT_PIPELINE_URL, files=files, data=data)
    if response.status_code == 200:
        result = response.json()
        display_pipeline_results(result)
    else:
        st.error(f"Pipeline processing filed: {response}")

def display_pipeline_results(result):
    st.success("Pipeline completed successfully!")
    input_image = Image.open(result["input_image"])
    input_image = input_image.resize((384, 384))
    st.image(input_image, caption="Input Image", width=384,  use_container_width =False)

    for stage_output in result["pipeline_results"]:
        st.subheader(f"Stage {stage_output['stage']} - {stage_output['model']}")
        st.image(stage_output["output_file_path"], width=384, use_container_width =False)

    st.subheader("Final Output")
    st.image(result["final_output"], caption="Final Output", width=384, use_container_width =False)

def display_outputs(outputs):
    if not outputs:
        st.info("No outputs available yet.")
        return

    # Get unique input images and sort them for consistent ordering
    unique_input_images = sorted({output["input_image"] for output in outputs})
    selected_image = st.session_state.get('selected_input_image')

    # Show either image grid OR predictions, not both
    if not selected_image:
        # Display image selection grid
        st.subheader("Select an Input Image")
        num_cols = 3  # Number of columns per row
        num_images = len(unique_input_images)

        for row_idx in range(0, num_images, num_cols):
            cols = st.columns(num_cols)
            for col_idx, col in enumerate(cols):
                img_idx = row_idx + col_idx
                if img_idx >= num_images:
                    break
                img_path = unique_input_images[img_idx]

                with col:
                    try:
                        img = Image.open(img_path)
                        img = img.resize((256, 256))
                        if st.button(f"Image {img_idx + 1}",
                                    key=f"btn_{img_path}",
                                    use_container_width=True):
                            st.session_state['selected_input_image'] = img_path
                            st.rerun()
                        col.image(img, use_container_width=True, caption=f"Image {img_idx + 1}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
    else:
        # Show predictions for selected image
        st.subheader("Selected Image Predictions")

        # Add back button at the top
        if st.button("← Back to Image Selection"):
            del st.session_state['selected_input_image']
            st.rerun()

        # Show full-size selected image
        try:
            full_img = Image.open(selected_image)
            st.image(full_img, use_container_width=False,width=384, caption=selected_image)
        except Exception as e:
            st.error(f"Error loading selected image: {str(e)}")

        # Filter controls
        pipeline_filter = st.selectbox(
            "Filter by Pipeline:",
            ["All", "deblur", "dehaze", "hybrid"],
            index=0,
            key="pipeline_filter"
        )

        # Get filtered predictions
        filtered_outputs = [
            output for output in outputs
            if output["input_image"] == selected_image and
               (pipeline_filter == "All" or output["pipeline"] == pipeline_filter)
        ]

        # Identify complete pipeline runs
        pipeline_runs = []
        visited = set()

        for output in filtered_outputs:
            if output["id"] in visited:
                continue

            # For hybrid pipelines, find complete sequences
            if output["pipeline"] == "hybrid":
                current = output
                stages = []

                # Find the root parent
                while True:
                    parent = next(
                        (o for o in filtered_outputs
                         if o["id"] == current["parent_model_output_id"]),
                        None
                    )
                    if not parent:
                        break
                    current = parent

                # Collect all children from root
                while current:
                    if current["id"] in visited:
                        break
                    stages.append(current)
                    visited.add(current["id"])
                    current = next(
                        (o for o in filtered_outputs
                         if o["parent_model_output_id"] == current["id"]),
                        None
                    )

                if stages:
                    pipeline_runs.append({
                        "stages": stages,
                        "timestamp": stages[0]["processed_at"],
                        "type": "hybrid"
                    })
            else:
                # Single-stage pipelines
                if output["id"] not in visited:
                    pipeline_runs.append({
                        "stages": [output],
                        "timestamp": output["processed_at"],
                        "type": output["pipeline"]
                    })
                    visited.add(output["id"])

        # Group by pipeline structure and keep only latest execution
        pipeline_groups = {}
        for run in pipeline_runs:
            stage_sequence = tuple([
                stage["pipeline_stage"].split("_")[-1]
                for stage in run["stages"]
            ])

            # Initialize group if it doesn't exist
            if stage_sequence not in pipeline_groups:
                pipeline_groups[stage_sequence] = {
                    "run": run,
                    "count": 1,
                    "first_seen": run["timestamp"],
                    "last_seen": run["timestamp"]
                }
            else:
                # Update existing group
                pipeline_groups[stage_sequence]["count"] += 1
                if run["timestamp"] < pipeline_groups[stage_sequence]["first_seen"]:
                    pipeline_groups[stage_sequence]["first_seen"] = run["timestamp"]
                if run["timestamp"] > pipeline_groups[stage_sequence]["last_seen"]:
                    pipeline_groups[stage_sequence]["last_seen"] = run["timestamp"]
                    pipeline_groups[stage_sequence]["run"] = run  # Update with latest run

        # Convert to sorted list
        pipeline_groups = sorted(
            pipeline_groups.items(),
            key=lambda x: x[1]["last_seen"],  # Sort by last execution time
            reverse=True
        )

        if not pipeline_groups:
            st.info("No predictions match the current filters.")
            return

        # Display results
        for (stage_sequence, group_data) in pipeline_groups:
            run = group_data["run"]
            formatted_stages = [
                f"S{i+1}: {stage.capitalize()}"
                for i, stage in enumerate(stage_sequence)
            ]

            with st.expander(
                f"{' → '.join(formatted_stages)} "
                f"(Executed {group_data['count']} times)",
                expanded=False
            ):
                st.caption(
                    f"First execution: {group_data['first_seen']} | "
                    f"Last execution: {group_data['last_seen']}"
                )

                # Show latest execution details
                for stage in run["stages"]:
                    with st.container():
                        cols = st.columns([1, 3])
                        with cols[0]:
                            parts = stage["pipeline_stage"].split('_')
                            stage_label = (
                                f"S{parts[1]}: {parts[-1].capitalize()}"
                                if len(parts) >= 2
                                else stage["pipeline_stage"].capitalize()
                            )
                            st.markdown(f"**{stage['pipeline'].capitalize()}**")
                            st.markdown(f"<small>{stage_label}</small>",
                                        unsafe_allow_html=True)

                        with cols[1]:
                            try:
                                result_img = Image.open(stage["output_file_path"])
                                st.image(result_img, use_container_width=False,width=384,
                                        caption=f"Stage result")
                            except Exception as e:
                                st.error(f"Error loading image: {str(e)}")
                        st.divider()

def global_outputs_page():
    st.header("Global Model Outputs")

    response = api_get(GLOBAL_OUTPUTS_URL)
    if response.status_code != 200:
        st.error(f"Failed to retrieve global outputs: {response.text}")
        return

    outputs = response.json()
    display_outputs(outputs)


def main():
    # Check if the user is authenticated
    token = get_token_from_url()
    if token:
        st.session_state["access_token"] = token
        response = requests.post(f"{BASE_URL}/user/detail/", headers={"token" : st.session_state["access_token"]})
        if response.status_code == 200:
            st.session_state["user_id"] = response.json().get("user_id")
        else:
            st.error("Authentication failed. Please log in again.")
            del st.session_state["access_token"]
            del st.query_params.token

    if "access_token" not in st.session_state:
        # Show Login/Register options as tabs in the main area
        auth_tab = st.tabs(["Login", "Register"])
        with auth_tab[0]:
            show_login_page()
        with auth_tab[1]:
            show_register_page()
    else:
        # Show main app menu in sidebar when logged in
        show_main_app()

if __name__ == "__main__":
    main()
