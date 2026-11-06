import unittest

from .gpu_guard import parse_graphics_processes


class GraphicsIsolationTests(unittest.TestCase):
    def test_egl_is_detected_even_without_compute_process(self):
        xml = '''<nvidia_smi_log><gpu><uuid>GPU-one</uuid><processes>
          <process_info><pid>123</pid><type>G</type><used_memory>537 MiB</used_memory></process_info>
          <process_info><pid>124</pid><type>C</type><used_memory>9000 MiB</used_memory></process_info>
          <process_info><pid>125</pid><type>C+G</type><used_memory>N/A</used_memory></process_info>
          </processes></gpu></nvidia_smi_log>'''
        self.assertEqual(parse_graphics_processes(xml, "GPU-one"),
                         [dict(pid=123, type="G", memory_mib=537.),
                          dict(pid=125, type="C+G", memory_mib=None)])

    def test_empty_process_list_is_allowed_but_wrong_gpu_is_not(self):
        xml = '<nvidia_smi_log><gpu><uuid>GPU-one</uuid><processes/></gpu></nvidia_smi_log>'
        self.assertEqual(parse_graphics_processes(xml, "GPU-one"), [])
        with self.assertRaises(RuntimeError):
            parse_graphics_processes(xml, "GPU-other")
