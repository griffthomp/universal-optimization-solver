#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DOUBLE BACKWARD DEBUGGER
==========================================
This script adds extensive logging to identify any remaining double backward issues.
It wraps all tensor operations with debug information.
"""

import torch
import functools
import traceback
from typing import Any, Callable

class DoubleBackwardDebugger:
    """Debug wrapper to catch double backward issues"""
    
    def __init__(self):
        self.tensor_operations = []
        self.backward_called = False
        self.operation_count = 0
    
    def log_tensor_op(self, operation_name: str, tensor: torch.Tensor, location: str):
        """Log tensor operation with context"""
        self.operation_count += 1
        
        op_info = {
            'id': self.operation_count,
            'operation': operation_name,
            'location': location,
            'requires_grad': tensor.requires_grad if hasattr(tensor, 'requires_grad') else False,
            'is_leaf': tensor.is_leaf if hasattr(tensor, 'is_leaf') else False,
            'grad_fn': str(tensor.grad_fn) if hasattr(tensor, 'grad_fn') and tensor.grad_fn else None,
            'shape': tuple(tensor.shape) if hasattr(tensor, 'shape') else None,
            'after_backward': self.backward_called,
            'stack_trace': ''.join(traceback.format_stack()[-3:-1])  # Get calling context
        }
        
        self.tensor_operations.append(op_info)
        
        # Warning for operations after backward
        if self.backward_called and tensor.requires_grad:
            print(f"⚠️  POTENTIAL ISSUE: {operation_name} on tensor with gradients after backward!")
            print(f"   Location: {location}")
            print(f"   Tensor grad_fn: {tensor.grad_fn}")
            print(f"   Stack trace: {op_info['stack_trace']}")
    
    def mark_backward_called(self):
        """Mark that backward has been called"""
        self.backward_called = True
        print(f"🔥 BACKWARD CALLED - Operation #{self.operation_count}")
    
    def reset_for_next_batch(self):
        """Reset for next batch"""
        self.backward_called = False
        print(f"🔄 RESET FOR NEXT BATCH - Total ops: {len(self.tensor_operations)}")
    
    def print_summary(self):
        """Print summary of all operations"""
        print(f"\n📊 TENSOR OPERATIONS SUMMARY:")
        print(f"   Total operations: {len(self.tensor_operations)}")
        
        after_backward = [op for op in self.tensor_operations if op['after_backward'] and op['requires_grad']]
        if after_backward:
            print(f"   ⚠️  Operations after backward with gradients: {len(after_backward)}")
            for op in after_backward:
                print(f"      - {op['operation']} at {op['location']}")
        else:
            print(f"   ✅ No problematic operations found")

# Global debugger instance
debugger = DoubleBackwardDebugger()

def debug_tensor_operation(operation_name: str, location: str = "unknown"):
    """Decorator to debug tensor operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Log if result is a tensor
            if isinstance(result, torch.Tensor):
                debugger.log_tensor_op(operation_name, result, location)
            
            return result
        return wrapper
    return decorator

def add_debugging_to_training_script():
    """Add debugging hooks to the training script"""
    
    # Read the training script
    with open('train_4090_ultra_scaled.py', 'r') as f:
        content = f.read()
    
    # Add debug imports at the top
    debug_imports = '''
# 🔍 DOUBLE BACKWARD DEBUGGING
import sys
sys.path.append('.')
from debug_double_backward import debugger, debug_tensor_operation

'''
    
    # Insert after existing imports
    import_end = content.find('def robust_json_parse')
    if import_end == -1:
        import_end = content.find('class GPT4oTeacher')
    
    content = content[:import_end] + debug_imports + content[import_end:]
    
    # Add debug hooks around critical operations
    replacements = [
        # Mark backward call
        ('loss.backward()', 'debugger.mark_backward_called(); loss.backward()'),
        
        # Debug tensor operations after backward
        ('actual_loss = loss.item()', 'debugger.log_tensor_op("loss.item()", loss, "actual_loss_extraction"); actual_loss = loss.item()'),
        
        # Debug accuracy calculation
        ('accuracy = (torch.abs(preds_detached - targets_detached)', 'debugger.log_tensor_op("accuracy_calc", preds_detached, "accuracy_calculation"); accuracy = (torch.abs(preds_detached - targets_detached)'),
        
        # Reset for next batch
        ('for batch_idx, batch in enumerate(dataloader):', 'for batch_idx, batch in enumerate(dataloader):\n        if batch_idx > 0: debugger.reset_for_next_batch()'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Added debug hook: {old[:30]}...")
    
    # Write the debugged version
    with open('train_4090_ultra_scaled_debug.py', 'w') as f:
        f.write(content)
    
    print("🔍 Created debug version: train_4090_ultra_scaled_debug.py")
    print("🚀 Run this version if the issue persists!")

if __name__ == "__main__":
    add_debugging_to_training_script() 