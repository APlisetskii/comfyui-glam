import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"
import { setWidgetConfig } from "/extensions/core/widgetInputs.js"

declare const LiteGraph: any

function createCallback(nodename: string, basename: string, inputType: string, withWeights?: string[]) {
    return async function (nodeType: any, nodeData: any, app: any) {
        if (nodeData.name !== nodename) {
            return
        }

        // Если это одна из нод "GlamSmoothZoom" или "GlamSmoothZoomOut",
        // пропускаем динамическую логику
        if (nodename === "GlamSmoothZoom" || nodename === "GlamSmoothZoomOut") {
            return
        }

        // Ниже остаётся логика для GlamRandomImage (динамические сокеты)
        const getInputBasename = (input: any) => input.name.split('_')[0]
        const getInputExtraname = (input: any) => input.name.split('_', 2)[1]

        const updateInputs = function (this: any) {
            for (let index = this.inputs.length; index--;) {
                const input = this.inputs[index]
                if (
                    getInputBasename(input) === basename &&
                    input.link === null &&
                    this.removeCancel !== index
                ) {
                    this.removeInput(index)
                    const widgetIndex = (this.widgets as any[]).findIndex(
                        (value) => value.name === input.name
                    )
                    if (widgetIndex !== -1) {
                        this.widgets.splice(widgetIndex, 1)
                    }
                }
            }
            let j = 0
            for (let i = 0, il = this.inputs.length; i < il; ++i) {
                const input = this.inputs[i]
                if (getInputBasename(input) === basename) {
                    this.inputs[i].name = [basename, j++].join('_')
                }
            }
            this.addInput([basename, j].join('_'), inputType)

            if (!this.widgets) {
                this.widgets = []
            }
            for (let i = 0, il = this.inputs.length; i < il; ++i) {
                const input = this.inputs[i]
                if (input.widget) {
                    setWidgetConfig(input, [input.type, { forceInput: true }])
                    continue
                }
                input.widget = { name: input.name }
                setWidgetConfig(input, [inputType, { forceInput: true }])
            }
        }

        const onNodeCreatedOriginal = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreatedOriginal) {
                const tmp = app.configuringGraph
                app.configuringGraph = false
                onNodeCreatedOriginal.call(this)
                app.configuringGraph = tmp
            }
            this.removeCancel = -1

            const onConnectInputOriginal = this.onConnectInput
            this.onConnectInput = function (
                targetSlot: number,
                type: string,
                output: any,
                originNode: any,
                originSlot: number
            ) {
                let retVal = onConnectInputOriginal
                    ? onConnectInputOriginal.apply(this, arguments)
                    : void 0
                if (
                    originNode.type === "PrimitiveNode" &&
                    getInputBasename(this.inputs[targetSlot]) === basename
                ) {
                    return false
                }
                this.removeCancel = targetSlot
                return retVal
            }

            const onInputDblClickOriginal = this.onInputDblClick
            this.onInputDblClick = function (slot: number) {
                if (onInputDblClickOriginal) {
                    const originalCreateNode = LiteGraph.createNode
                    if (getInputBasename(this.inputs[slot]) === basename) {
                        LiteGraph.createNode = function (nodeType: string) {
                            if (nodeType !== "PrimitiveNode") {
                                return originalCreateNode.apply(this, arguments)
                            }
                            return originalCreateNode.call(this, "StringToolsText")
                        }
                    }
                    onInputDblClickOriginal.call(this, slot)
                    LiteGraph.createNode = originalCreateNode
                }
            }

            const onConnectionsChange = this.onConnectionsChange
            this.onConnectionsChange = function (type: number, slotIndex: number, isConnected: boolean) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments)
                }
                if (type !== 1) {
                    return
                }
                updateInputs.call(this)
                this.removeCancel = -1
            }

            this.onAdded = function (graph: any) {
                this.tmpWidgets = this.widgets
                if (app.configuringGraph) {
                    this.widgets = []
                    this.widgets_values = []
                } else {
                    updateInputs.call(this)
                }
            }

            const onGraphConfigured = this.onGraphConfigured
            this.onGraphConfigured = function () {
                if (this.tmpWidgets) {
                    this.widgets = this.tmpWidgets.concat(this.widgets)
                    delete this.tmpWidgets
                }
                if (app.configuringGraph) {
                    updateInputs.call(this)
                }
            }

            if (withWeights !== void 0) {
                this.calcNodeInputs = function (prompt: any, workflow: any) {
                    // ... логика withWeights ...
                }
            }

            if (!this.inputs) {
                this.inputs = []
            }
        }
    }
}

const queuePromptOriginal = api.queuePrompt
api.queuePrompt = (async function queuePrompt(number: number, { output, workflow }: { output: any, workflow: any }) {
    for (const id of Object.keys(output)) {
        const node = app.graph.getNodeById(id)
        if (node.calcNodeInputs && typeof node.calcNodeInputs === "function") {
            node.calcNodeInputs(output, workflow)
        }
    }
    return await queuePromptOriginal(number, { output, workflow })
}).bind(api)

// Регистрируем 3 ноды:
// 1) GlamRandomImage (много входов)
// 2) GlamSmoothZoom (один вход)
// 3) GlamSmoothZoomOut (один вход)
app.registerExtension({
    name: "Taremin.GlamRandomImage",
    beforeRegisterNodeDef: createCallback("GlamRandomImage", "image", "IMAGE"),
})

app.registerExtension({
    name: "Taremin.GlamSmoothZoom",
    beforeRegisterNodeDef: createCallback("GlamSmoothZoom", "image", "IMAGE"),
})

app.registerExtension({
    name: "Taremin.GlamSmoothZoomOut",
    beforeRegisterNodeDef: createCallback("GlamSmoothZoomOut", "image", "IMAGE"),
})
