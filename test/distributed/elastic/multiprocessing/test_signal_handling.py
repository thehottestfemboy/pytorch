#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import sys
import time
import unittest
from unittest.mock import patch

import torch.distributed.elastic.multiprocessing.api as api
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from torch.distributed.elastic.multiprocessing.api import (
    SignalConfig,
    configure_signal_handlers,
    MultiprocessContext,
    SubprocessContext,
)

def sleep_fn():
    time.sleep(10)

def signal_handler_fn(signum, frame):
    sys.exit(0)

class SignalHandlingTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "signal_test_dir")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_signal_config(self):
        """Test that signal configuration works as expected"""
        # Test default values
        config = SignalConfig()
        self.assertEqual(30, config.grace_period)
        self.assertFalse(config.handle_sigusr)
        self.assertTrue(config.forward_signals)

        # Test custom values
        config = SignalConfig(grace_period=60, handle_sigusr=True, forward_signals=False)
        self.assertEqual(60, config.grace_period)
        self.assertTrue(config.handle_sigusr)
        self.assertFalse(config.forward_signals)

    def test_configure_signal_handlers(self):
        """Test that configure_signal_handlers updates global config"""
        configure_signal_handlers(grace_period=45, handle_sigusr=True, forward_signals=False)
        self.assertEqual(45, api.SIGNAL_CONFIG.grace_period)
        self.assertTrue(api.SIGNAL_CONFIG.handle_sigusr)
        self.assertFalse(api.SIGNAL_CONFIG.forward_signals)

    @unittest.skipIf(sys.platform == "win32", "SIGUSR signals not supported on Windows")
    def test_sigusr_handling(self):
        """Test that SIGUSR signals are handled when enabled"""
        configure_signal_handlers(handle_sigusr=True)
        
        # Start a process that sleeps
        pc = MultiprocessContext(
            name="test_sigusr",
            entrypoint=sleep_fn,
            args={0: ()},
            envs={0: {}},
            start_method="spawn",
            logs_specs=api.DefaultLogsSpecs(log_dir=self.test_dir),
        )
        pc.start()
        
        # Send SIGUSR1 to the process
        pid = pc.pids()[0]
        os.kill(pid, signal.SIGUSR1)
        
        # Process should exit with signal exception
        result = pc.wait(timeout=5)
        self.assertTrue(result.is_failed())
        self.assertIn(0, result.failures)
        failure = result.failures[0]
        self.assertEqual(pid, failure.pid)

    def test_grace_period(self):
        """Test that grace period is respected before SIGKILL"""
        configure_signal_handlers(grace_period=2)
        
        # Start a process that ignores SIGTERM
        pc = MultiprocessContext(
            name="test_grace",
            entrypoint=signal_handler_fn,
            args={0: ()},
            envs={0: {}},
            start_method="spawn",
            logs_specs=api.DefaultLogsSpecs(log_dir=self.test_dir),
        )
        pc.start()
        
        start_time = time.time()
        pc.close()  # This should wait grace_period before SIGKILL
        end_time = time.time()
        
        # Should have waited at least grace_period seconds
        self.assertGreaterEqual(end_time - start_time, 2)

    def test_no_signal_forwarding(self):
        """Test that signals are not forwarded when disabled"""
        configure_signal_handlers(forward_signals=False)
        
        with patch('os.kill') as mock_kill:
            pc = MultiprocessContext(
                name="test_no_forward",
                entrypoint=sleep_fn,
                args={0: ()},
                envs={0: {}},
                start_method="spawn",
                logs_specs=api.DefaultLogsSpecs(log_dir=self.test_dir),
            )
            pc.start()
            pc.close()
            
            # os.kill should not have been called since forwarding is disabled
            mock_kill.assert_not_called()

if __name__ == '__main__':
    unittest.main() 