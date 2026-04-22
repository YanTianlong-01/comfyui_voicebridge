import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

/**
 * Wire up an "upload .srt" button next to the `srt` dropdown widget
 * on VoiceBridge's LoadSRT node, mirroring the Load Audio / Load Image UX.
 *
 * Uploaded files land in ComfyUI/input/ via the standard /upload/image
 * endpoint (which accepts arbitrary file types despite the name), and the
 * new filename is appended to the dropdown so the user can select it.
 */
function srtUpload(node, inputName, inputData, app) {
    const srtWidget = node.widgets.find((w) => w.name === "srt");
    let uploadWidget;

    const default_value = srtWidget.value;
    Object.defineProperty(srtWidget, "value", {
        set: function (value) {
            this._real_value = value;
        },
        get: function () {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_value;
            }

            if (value.filename) {
                const real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }
                value += real_value.filename;
                if (real_value.type && real_value.type !== "input") {
                    value += ` [${real_value.type}]`;
                }
            }
            return value;
        },
    });

    async function uploadFile(file, updateNode, pasted = false) {
        try {
            const body = new FormData();
            body.append("image", file);
            if (pasted) body.append("subfolder", "pasted");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body,
            });

            if (resp.status === 200) {
                const data = await resp.json();
                let path = data.name;
                if (data.subfolder) path = data.subfolder + "/" + path;

                if (!srtWidget.options.values.includes(path)) {
                    srtWidget.options.values.push(path);
                }
                if (updateNode) {
                    srtWidget.value = path;
                }
            } else {
                alert(resp.status + " - " + resp.statusText);
            }
        } catch (error) {
            alert(error);
        }
    }

    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: ".srt,.txt,text/plain",
        style: "display: none",
        onchange: async () => {
            if (fileInput.files.length) {
                await uploadFile(fileInput.files[0], true);
            }
        },
    });
    document.body.append(fileInput);

    uploadWidget = node.addWidget("button", "choose srt file to upload", "SRT", () => {
        fileInput.click();
    });
    uploadWidget.serialize = false;

    const cb = node.callback;
    srtWidget.callback = function () {
        if (cb) {
            return cb.apply(this, arguments);
        }
    };

    return { widget: uploadWidget };
}

// Register the widget type with ComfyUI
ComfyWidgets.SRTPLOAD = srtUpload;

// Hook into the node definition right before ComfyUI registers it,
// injecting an extra `upload` input of our custom widget type.
app.registerExtension({
    name: "VoiceBridge.UploadSRT",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match the node class name registered in NODE_CLASS_MAPPINGS
        if (nodeData?.name === "VoiceBridgeLoadSRT") {
            nodeData.input.required.upload = ["SRTPLOAD"];
        }
    },
});
