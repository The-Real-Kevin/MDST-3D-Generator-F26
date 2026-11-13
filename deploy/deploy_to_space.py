"""
Upload the assembled Space folder to Hugging Face.

    HF_TOKEN=... HF_SPACE=user/space-name python deploy/deploy_to_space.py space

Used by .github/workflows/deploy-space.yml, but works by hand too. It goes
through the Hugging Face upload API rather than `git push`: the Hub rejects
binary files (model.pt, the .png previews) pushed with plain git, and the
API stores them properly without any git-lfs setup.
"""
import os
import sys

from huggingface_hub import HfApi

folder = sys.argv[1] if len(sys.argv) > 1 else "space"
space = os.environ.get("HF_SPACE", "").strip()
token = os.environ.get("HF_TOKEN", "").strip()
if not space or not token:
    sys.exit("Set HF_SPACE (e.g. your-username/mdst-city-generator) and HF_TOKEN (a WRITE token).")

api = HfApi(token=token)
api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
sha = os.environ.get("GITHUB_SHA", "")[:7]
api.upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=folder,
    commit_message=f"Deploy {sha} from GitHub" if sha else "Deploy from local machine",
    # mirror the code: files deleted on GitHub are deleted on the Space too
    # (only inside these paths -- the Space's own .gitattributes is left alone)
    delete_patterns=["app.py", "requirements.txt", "citygen/*", "scripts/*", "text2city/*", "data/*"],
)
print(f"deployed -> https://huggingface.co/spaces/{space}")
