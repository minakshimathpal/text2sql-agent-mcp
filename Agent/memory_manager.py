"""
Memory management utilities for the text2sql agent.
"""
import gc
import os
import time
import psutil
import threading
import weakref
from contextlib import contextmanager
from typing import Optional, Callable, Dict, Any

class MemoryManager:
    """Memory manager for monitoring and managing system memory."""
    
    def __init__(self, threshold_mb: float = 1000.0, critical_mb: float = 300.0,
                 check_interval: float = 0.5, monitor_active: bool = True):
        """Initialize the memory manager.

        Args:
            threshold_mb: Memory threshold in MB below which to take action
            critical_mb: Critical memory level in MB below which to abort operations
            check_interval: Interval in seconds for background memory checks
            monitor_active: Whether to start the background monitor thread
        """
        # Allow environment overrides to tune aggressiveness
        try:
            _env_threshold = os.environ.get("MEM_THRESHOLD_MB")
            _env_critical = os.environ.get("MEM_CRITICAL_MB")
            if _env_threshold: threshold_mb = float(_env_threshold)
            if _env_critical: critical_mb = float(_env_critical)
        except Exception:
            pass
        self.threshold_mb = threshold_mb
        self.critical_mb = critical_mb
        self.check_interval = check_interval
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._keep_monitoring = False

        # Track instances with weak references when possible to avoid holding
        # strong references that prevent GC. Keep a fallback list for objects
        # that are not weak-referenceable.
        self._active_llm_refs = []  # list of weakref.ref where possible
        self._active_llm_strong = []  # fallback strong refs
        self._instance_count = 0

        # Soft cap to avoid accidentally registering many parallel LLMs
        self.max_instances = 1

        # Margin (MB) above critical required to allow new registrations
        self.registration_margin_mb = 200.0

        # Track last emergency cleanup time to avoid repeated rapid-fire releases
        self._last_emergency = 0.0
        self._emergency_cooldown = 10.0  # seconds between emergency attempts
        # Counter-based suspension flag to prevent emergency releases while in critical sections
        self._suspend_emergency_depth = 0

        # Minimum available memory (MB) at which we will STILL retain a single LLM instance
        # even if below critical, to avoid thrashing (re-init cost is high). Configurable.
        try:
            self.min_retain_mb = float(os.environ.get("MIN_RETAIN_LLM_MB", "250"))
        except Exception:
            self.min_retain_mb = 250.0

        # Start memory monitor if requested
        if monitor_active:
            self.start_monitor()

    def _prune_active_instances(self):
        """Remove dead weakrefs and return list of live instances."""
        live = []
        # Prune weak refs
        for ref in list(self._active_llm_refs):
            inst = ref()
            if inst is None:
                try:
                    self._active_llm_refs.remove(ref)
                except ValueError:
                    pass
            else:
                live.append(inst)
        # Add strong refs
        for inst in list(self._active_llm_strong):
            if inst is not None:
                live.append(inst)
        return live
    
    def register_llm_instance(self, llm_instance):
        """Register an LLM instance for tracking.

        This will refuse registration (and attempt to release the provided
        instance) if the system memory is too low. It prefers weak references
        so that the manager itself does not keep objects alive.
        """
        with self._lock:
            # Prune dead weakrefs first
            self._prune_active_instances()

            stats = self.get_memory_stats()
            # Relax policy: always allow the FIRST instance as long as we are above critical.
            # Only enforce margin for additional instances (even though max_instances is 1 today).
            if self._instance_count == 0:
                if stats['available_mb'] < self.critical_mb:
                    # Too dangerous even for first instance
                    try:
                        if hasattr(llm_instance, 'close'):
                            llm_instance.close()
                        elif hasattr(llm_instance, 'release'):
                            llm_instance.release()
                    except Exception:
                        pass
                    raise MemoryError(f"Refusing first LLM registration: below critical ({stats['available_mb']:.1f}MB < {self.critical_mb}MB)")
            else:
                # For subsequent instances keep the stricter margin
                if stats['available_mb'] < (self.critical_mb + self.registration_margin_mb):
                    try:
                        if hasattr(llm_instance, 'close'):
                            llm_instance.close()
                        elif hasattr(llm_instance, 'release'):
                            llm_instance.release()
                    except Exception:
                        pass
                    raise MemoryError(f"Refusing to register additional LLM (avail={stats['available_mb']:.1f}MB; need >= {self.critical_mb + self.registration_margin_mb:.1f}MB)")

            # Enforce max instances cap
            if self._instance_count >= self.max_instances:
                raise MemoryError(f"Refusing to register LLM instance: max_instances={self.max_instances} reached")

            # Try to create a weakref to avoid holding a strong reference
            try:
                ref = weakref.ref(llm_instance, lambda r: None)
                self._active_llm_refs.append(ref)
            except TypeError:
                # Object not weak-referenceable; store strong ref but warn
                self._active_llm_strong.append(llm_instance)

            self._instance_count += 1
            print(f"LLM instance registered (total: {self._instance_count})")
    
    def unregister_llm_instance(self, llm_instance):
        """Unregister an LLM instance from tracking."""
        with self._lock:
            # Remove from weak refs
            removed = False
            for ref in list(self._active_llm_refs):
                inst = ref()
                if inst is None:
                    # Dead ref; prune
                    try:
                        self._active_llm_refs.remove(ref)
                    except ValueError:
                        pass
                    continue
                if inst is llm_instance:
                    try:
                        self._active_llm_refs.remove(ref)
                    except ValueError:
                        pass
                    removed = True
                    break

            # Remove from strong refs
            if not removed:
                for i, instance in enumerate(list(self._active_llm_strong)):
                    if instance is llm_instance:
                        del self._active_llm_strong[i]
                        removed = True
                        break

            if removed:
                self._instance_count = max(0, self._instance_count - 1)
                print(f"LLM instance unregistered (remaining: {self._instance_count})")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total / (1024 * 1024),
            "available_mb": vm.available / (1024 * 1024),
            "used_mb": vm.used / (1024 * 1024),
            "percent_used": vm.percent
        }
    
    def is_memory_safe(self) -> bool:
        """Check if memory is at a safe level."""
        return self.get_memory_stats()["available_mb"] >= self.threshold_mb
    
    def is_memory_critical(self) -> bool:
        """Check if memory is at a critical level."""
        stats = self.get_memory_stats()
        available = stats["available_mb"]
        is_critical = available <= self.critical_mb
        
        # Debug logging for memory state
        if is_critical:
            print(f"🚨 [MEMORY_DEBUG] Critical state: {available:.1f}MB <= {self.critical_mb}MB threshold")
        
        return is_critical
    
    def force_collect_garbage(self):
        """Force aggressive garbage collection."""
        # Run multiple collection cycles
        for i in range(3):
            gc.collect(i)
        
        # Clear any reference cycles
        if hasattr(gc, 'freeze'):
            gc.freeze()
        
        # Clear module caches if possible
        if hasattr(gc, 'clear_caches'):
            gc.clear_caches()
            
        # Log memory status after cleanup
        stats = self.get_memory_stats()
        #print(f"After garbage collection: {stats['available_mb']:.2f}MB available ({stats['percent_used']}% used)")
    
    def _monitor_memory(self):
        """Background thread for memory monitoring."""
        while self._keep_monitoring:
            stats = self.get_memory_stats()
            
            # Check for critical memory conditions
            if stats["available_mb"] <= self.critical_mb:
                #print(f"⚠️ CRITICAL MEMORY WARNING: Only {stats['available_mb']:.2f}MB available!")
                self.force_collect_garbage()
                
                # If still critical after cleanup, take more drastic measures
                if self.is_memory_critical():
                    now = time.time()
                    # Only attempt emergency release if cooldown passed
                    if now - self._last_emergency > self._emergency_cooldown:
                        # Check if we have tracked instances (prune dead refs first)
                        live = self._prune_active_instances()
                        if live:
                            # If only a single LLM instance and we still have above min_retain_mb, skip releasing to prevent constant re-init.
                            if len(live) == 1 and stats["available_mb"] > self.min_retain_mb:
                                print(f"⚠️ Critical memory but retaining single LLM (avail={stats['available_mb']:.1f}MB > {self.min_retain_mb}MB safeguard)")
                            else:
                                # Skip emergency release while suspended
                                if self._suspend_emergency_depth > 0:
                                    print(f"🚫 Emergency release suppressed (depth={self._suspend_emergency_depth}) during critical section")
                                else:
                                    self._emergency_release_llm_instances()
                        else:
                            # Log once per cooldown when no instances are registered
                            print("🚨 Emergency memory state detected but no LLM instances registered to release")
                        self._last_emergency = now
                    else:
                        # Cooldown active; skip repeated emergency attempts
                        pass
            
            # Regular check for low memory
            elif stats["available_mb"] <= self.threshold_mb:
                #print(f"⚠️ Low memory warning: {stats['available_mb']:.2f}MB available")
                self.force_collect_garbage()
            
            # Sleep for the check interval
            time.sleep(self.check_interval)
    
    def _emergency_release_llm_instances(self):
        """Emergency release of LLM instances to free memory."""
        with self._lock:
            # Build list of live instances from weak and strong refs
            instances = self._prune_active_instances()
            print(f"Attempting to release {len(instances)} LLM instances")

            for llm in instances:
                try:
                    # Try to close or release resources
                    if hasattr(llm, 'close'):
                        llm.close()
                    elif hasattr(llm, 'release'):
                        llm.release()
                    
                    # Set to None to break references
                    if hasattr(llm, '_client'):
                        llm._client = None
                    if hasattr(llm, '_model'):
                        llm._model = None
                except Exception as e:
                    print(f"Error releasing LLM instance: {str(e)}")
            
            # Clear our tracking lists
            self._active_llm_refs.clear()
            self._active_llm_strong.clear()
            self._instance_count = 0

            # Force garbage collection after cleanup and attempt to return pages
            self.force_collect_garbage()
            try:
                # Best-effort working set trim to return pages to OS (Windows)
                self.trim_working_set()
            except Exception:
                pass

    # ---------------- Public high-level cleanup APIs -----------------
    def release_all_llms(self):
        """Public wrapper to release all tracked LLM instances (non-emergency manual call)."""
        self._emergency_release_llm_instances()

    def trim_working_set(self):
        """Attempt to return unused pages to OS (best-effort, platform-specific)."""
        try:
            import platform, ctypes
            if platform.system().lower() == 'windows':
                PROCESS_SET_QUOTA = 0x0100
                PROCESS_QUERY_INFORMATION = 0x0400
                kernel32 = ctypes.windll.kernel32
                psapi = ctypes.windll.psapi
                GetCurrentProcess = kernel32.GetCurrentProcess
                hProc = GetCurrentProcess()
                psapi.EmptyWorkingSet(hProc)
            else:
                # For *nix, reading /proc/self might encourage trimming after gc
                pass
        except Exception as e:
            print(f"Working set trim not supported/failed: {e}")

    def perform_full_cleanup(self) -> Dict[str, float]:
        """Comprehensive cleanup: release LLMs, GC, trim working set, return new stats."""
        before = self.get_memory_stats()
        self.release_all_llms()
        self.force_collect_garbage()
        self.trim_working_set()
        after = self.get_memory_stats()
        delta = after['available_mb'] - before['available_mb']
        print(f"Post-query full cleanup reclaimed {delta:.2f}MB (avail: {after['available_mb']:.2f}MB)")
        return after

    # --------------- Critical section helpers ---------------
    @contextmanager
    def suspend_emergency(self, reason: str = ""):
        """Context manager to temporarily suspend emergency LLM releases.

        Use around critical sections (e.g., active LLM generation) to prevent
        the monitor thread from tearing down the only LLM mid-call, which can
        cause intermittent generation failures.
        """
        with self._lock:
            self._suspend_emergency_depth += 1
            if reason:
                print(f"⏸️ Suspended emergency releases (depth={self._suspend_emergency_depth}) reason={reason}")
        try:
            yield
        finally:
            with self._lock:
                self._suspend_emergency_depth = max(0, self._suspend_emergency_depth - 1)
                if reason:
                    print(f"▶️ Resumed emergency releases (depth={self._suspend_emergency_depth}) reason={reason}")
    
    def start_monitor(self):
        """Start the background memory monitor."""
        with self._lock:
            if self._monitor_thread is None or not self._monitor_thread.is_alive():
                self._keep_monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_memory,
                    daemon=True,
                    name="MemoryMonitorThread"
                )
                self._monitor_thread.start()
                print("Memory monitor started in background")
    
    def stop_monitor(self):
        """Stop the background memory monitor."""
        with self._lock:
            self._keep_monitoring = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
                print("Memory monitor stopped")

# Create a global memory manager instance
memory_manager = MemoryManager(
    threshold_mb=500.0,   # 500MB — warning/action threshold (reduced from 1000MB)
    critical_mb=300.0,    # 300MB — BSOD-protection hard floor (unchanged)
    check_interval=1.0,   # Check every second
    monitor_active=True   # Start monitoring immediately
)

class TimeoutManager:
    """Manages operation timeouts with proper cleanup."""
    
    @staticmethod
    def run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Run a function with a timeout.
        
        Args:
            func: Function to run
            timeout: Timeout in seconds
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the function
            
        Raises:
            TimeoutError: If the function times out
        """
        result = None
        exception = None
        completed = threading.Event()
        
        # Thread function
        def worker():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
            finally:
                completed.set()
        
        # Start the worker thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        # Wait for completion or timeout
        if not completed.wait(timeout):
            # Force garbage collection before raising timeout
            memory_manager.force_collect_garbage()
            
            # Raise timeout error
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # If there was an exception, raise it
        if exception is not None:
            raise exception
        
        return result

def safe_check_memory(threshold_mb: Optional[float] = None) -> bool:
    """
    Safely check if there's enough memory available.
    Also performs garbage collection if memory is low.
    
    Args:
        threshold_mb: Optional threshold in MB (uses memory manager's threshold if None)
        
    Returns:
        True if memory is safe, False if it's below threshold
    """
    # Use memory manager's threshold if none provided
    if threshold_mb is None:
        threshold_mb = memory_manager.threshold_mb
    
    # Get current memory stats
    stats = memory_manager.get_memory_stats()
    
    # Log memory status
    print(f"Memory status: {stats['available_mb']:.2f}MB available of {stats['total_mb']:.2f}MB total")
    
    # If memory is below threshold, run garbage collection
    if stats['available_mb'] < threshold_mb:
        memory_manager.force_collect_garbage()
        
        # Get updated stats after garbage collection
        stats = memory_manager.get_memory_stats()
        
        # If still below threshold, return False
        if stats['available_mb'] < threshold_mb:
            print(f"⚠️ Low memory warning: Only {stats['available_mb']:.2f}MB available, need at least {threshold_mb:.2f}MB")
            return False
    
    return True
