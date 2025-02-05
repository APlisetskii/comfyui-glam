import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { setWidgetConfig } from "/extensions/core/widgetInputs.js";

function createCallback(nodename, basename, inputType, withWeights) {
    return async function (nodeType, nodeData, app) {
        if (nodeData.name !== nodename) {
            return;
        }

        // Если нода = GlamSmoothZoom или GlamSmoothZoomOut -> не делаем динамические входы
        if (nodename === "GlamSmoothZoom" || nodename === "GlamSmoothZoomOut") {
            return;
        }

        // ... остальная логика dynamic sockets (GlamRandomImage) ...
    };
}

// Переопределяем queuePrompt, если нужно
const queuePromptOriginal = api.queuePrompt;
api.queuePrompt = (async function queuePrompt(number, { output, workflow }) {
    for (const id of Object.keys(output)) {
        const node = app.graph.getNodeById(id);
        if (node.calcNodeInputs && typeof node.calcNodeInputs === "function") {
            node.calcNodeInputs(output, workflow);
        }
    }
    return await queuePromptOriginal(number, { output, workflow });
}).bind(api);

// Регистрируем 3 ноды
app.registerExtension({
    name: "Taremin.GlamRandomImage",
    beforeRegisterNodeDef: createCallback("GlamRandomImage", "image", "IMAGE"),
});

app.registerExtension({
    name: "Taremin.GlamSmoothZoom",
    beforeRegisterNodeDef: createCallback("GlamSmoothZoom", "image", "IMAGE"),
});

// Новая нода для Zoom Out
app.registerExtension({
    name: "Taremin.GlamSmoothZoomOut",
    beforeRegisterNodeDef: createCallback("GlamSmoothZoomOut", "image", "IMAGE"),
});
