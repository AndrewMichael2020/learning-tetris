// Activity Logger Module
class ActivityLogger {
    constructor(logElementId) {
        this.logElement = document.getElementById(logElementId);
        this.maxLogEntries = 100; // Limit log entries to prevent memory issues
    }

    logActivity(icon, category, message, details = '') {
        if (!this.logElement) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'activity-entry';
        
        const detailsHtml = details ? `<div class="activity-details">${details}</div>` : '';
        
        logEntry.innerHTML = `
            <div class="activity-header">
                <span class="activity-icon">${icon}</span>
                <span class="activity-category">${category}</span>
                <span class="activity-timestamp">${timestamp}</span>
            </div>
            <div class="activity-message">${message}</div>
            ${detailsHtml}
        `;
        
        this.logElement.insertBefore(logEntry, this.logElement.firstChild);
        
        // Clean up old entries
        this.limitLogEntries();
        
        console.log(`[${category}] ${message}`);
    }

    limitLogEntries() {
        while (this.logElement.children.length > this.maxLogEntries) {
            this.logElement.removeChild(this.logElement.lastChild);
        }
    }

    clear() {
        if (this.logElement) {
            this.logElement.innerHTML = '';
        }
    }

    // Specialized logging methods
    logError(message, details = '') {
        this.logActivity('❌', 'ERROR', message, details);
    }

    logWarning(message, details = '') {
        this.logActivity('⚠️', 'WARNING', message, details);
    }

    logSuccess(message, details = '') {
        this.logActivity('✅', 'SUCCESS', message, details);
    }

    logInfo(message, details = '') {
        this.logActivity('ℹ️', 'INFO', message, details);
    }

    logTraining(message, details = '') {
        this.logActivity('🎯', 'TRAINING', message, details);
    }

    logWebSocket(message, details = '') {
        this.logActivity('🔌', 'WEBSOCKET', message, details);
    }

    logSettings(message, details = '') {
        this.logActivity('🔧', 'SETTINGS', message, details);
    }
}
