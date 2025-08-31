// WebSocket Management Module
class WebSocketManager {
    constructor() {
        this.websockets = {};
        this.activeOperation = null;
    }

    createWebSocket(url, type) {
        if (this.websockets[type]) {
            this.websockets[type].close();
        }
        this.websockets[type] = new WebSocket(url);
        return this.websockets[type];
    }

    closeWebSocket(type) {
        if (this.websockets[type]) {
            this.websockets[type].close();
            delete this.websockets[type];
        }
    }

    closeAll() {
        Object.keys(this.websockets).forEach(type => {
            this.closeWebSocket(type);
        });
    }

    setActiveOperation(operation) {
        this.activeOperation = operation;
    }

    clearActiveOperation() {
        if (this.activeOperation !== null) {
            this.activeOperation = null;
            return true; // Indicate operation was cleared
        }
        return false; // No operation was active
    }

    getActiveOperation() {
        return this.activeOperation;
    }
}
