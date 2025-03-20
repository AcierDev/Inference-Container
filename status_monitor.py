# status_monitor.py
import curses
import queue
import threading
import time
from datetime import datetime
from collections import deque
from typing import Optional, Deque, Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

class StatusType(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class LogEntry:
    timestamp: datetime
    status_type: StatusType
    message: str

class StatusMonitor:
    def __init__(self, max_logs: int = 100):
        self.logs: Deque[LogEntry] = deque(maxlen=max_logs)
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'current_status': 'Idle',
            'start_time': datetime.now(),
            'last_error': None
        }
        self.running = False
        self._lock = threading.Lock()
        self._screen: Optional[curses.window] = None
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start the status monitor with proper error handling"""
        try:
            self.running = True
            self._screen = curses.initscr()
            
            # Initialize colors
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_RED, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            
            # Configure terminal
            curses.noecho()
            curses.cbreak()
            self._screen.keypad(True)
            
            # Start display thread
            self.display_thread = threading.Thread(
                target=self._display_loop,
                name="StatusMonitorDisplay"
            )
            self.display_thread.daemon = True
            self.display_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start status monitor: {str(e)}", exc_info=True)
            self.stop()
            raise

    def stop(self) -> None:
        """Safely stop the status monitor"""
        self.running = False
        
        if hasattr(self, 'display_thread'):
            try:
                self.display_thread.join(timeout=1.0)
            except threading.TimeoutError:
                self.logger.warning("Display thread did not stop cleanly")

        if self._screen:
            try:
                curses.nocbreak()
                self._screen.keypad(False)
                curses.echo()
                curses.endwin()
            except Exception as e:
                self.logger.error(f"Error during curses cleanup: {str(e)}")

    def update_status(self, message: str, status_type: str) -> None:
        """Update status with thread-safe operations"""
        try:
            status_type = StatusType(status_type)
            timestamp = datetime.now()
            
            with self._lock:
                self.logs.append(LogEntry(timestamp, status_type, message))
                
                if status_type == StatusType.SUCCESS:
                    self.stats['successful'] += 1
                    self.stats['total_processed'] += 1
                elif status_type == StatusType.ERROR:
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
                    self.stats['last_error'] = timestamp
                
                self.stats['current_status'] = message
                
        except Exception as e:
            self.logger.error(f"Error updating status: {str(e)}", exc_info=True)

    def _draw_header(self, max_x: int) -> int:
        """Draw the header section"""
        header = " Wood Imperfection Detection Monitor "
        self._screen.addstr(0, (max_x - len(header)) // 2, header, curses.A_BOLD)
        return 2

    def _draw_stats(self, y: int, max_x: int) -> int:
        """Draw the statistics section"""
        with self._lock:
            uptime = datetime.now() - self.stats['start_time']
            uptime_str = f"Uptime: {str(uptime).split('.')[0]}"
            
            stats = [
                f"Processed: {self.stats['total_processed']}",
                f"Successful: {self.stats['successful']}",
                f"Failed: {self.stats['failed']}",
                uptime_str
            ]
            
            stats_str = " | ".join(stats)
            self._screen.addstr(y, (max_x - len(stats_str)) // 2, stats_str)
            
            # Draw current status
            status_str = f"Current Status: {self.stats['current_status']}"
            self._screen.addstr(y + 2, 2, status_str)
            
            if self.stats['last_error']:
                error_str = f"Last Error: {self.stats['last_error'].strftime('%Y-%m-%d %H:%M:%S')}"
                self._screen.addstr(y + 3, 2, error_str, curses.color_pair(2))
                return 4
            return 3

    def _draw_logs(self, start_y: int, max_y: int, max_x: int) -> None:
        """Draw the log entries"""
        self._screen.addstr(start_y, 2, "Recent Activity:", curses.A_BOLD)
        log_y = start_y + 1
        
        color_map = {
            StatusType.SUCCESS: curses.color_pair(1),
            StatusType.ERROR: curses.color_pair(2),
            StatusType.WARNING: curses.color_pair(3),
            StatusType.INFO: curses.color_pair(4)
        }
        
        for log_entry in list(self.logs)[-10:]:
            if log_y >= max_y - 2:
                break
                
            timestamp_str = log_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            log_str = f"{timestamp_str} - {log_entry.message}"
            
            if len(log_str) > max_x - 4:
                log_str = log_str[:max_x - 7] + "..."
                
            self._screen.addstr(log_y, 2, log_str, color_map[log_entry.status_type])
            log_y += 1

    def _display_loop(self) -> None:
        """Main display loop with improved error handling and layout"""
        while self.running:
            try:
                self._screen.clear()
                max_y, max_x = self._screen.getmaxyx()
                
                # Draw sections
                current_y = self._draw_header(max_x)
                current_y += self._draw_stats(current_y, max_x) + 1
                self._draw_logs(current_y, max_y, max_x)
                
                # Draw footer
                footer = " Press 'q' to quit "
                self._screen.addstr(max_y - 1, (max_x - len(footer)) // 2, 
                                  footer, curses.A_BOLD)
                
                self._screen.refresh()
                
                # Check for 'q' key press
                self._screen.timeout(100)
                try:
                    if self._screen.getch() == ord('q'):
                        self.running = False
                except curses.error:
                    pass
                    
            except curses.error:
                continue
            except Exception as e:
                self.logger.error(f"Error in display loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on persistent errors