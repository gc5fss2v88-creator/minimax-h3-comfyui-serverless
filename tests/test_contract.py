import base64, json, pathlib, unittest
import handler

class ContractTests(unittest.TestCase):
    def test_defaults_and_frames(self):
        p=handler._params({'params':{'prompt':'test'}})
        self.assertEqual(p['steps'],8); self.assertEqual(p['fps'],24); self.assertEqual(handler._frames(15),362)
    def test_workflow_injects_fp8_params(self):
        inp={'params':{'prompt':'a test','steps':6,'seed':7},'images':[{'name':'x.png','data':base64.b64encode(b'not-an-image').decode()}]}
        old=handler._upload; handler._upload=lambda item:'x.png'
        try:
            wf=handler._workflow(inp,handler._params(inp)); self.assertEqual(wf['unet']['inputs']['unet_name'],'minimax_h3_fl2va_pruned_fp8_scaled.safetensors'); self.assertEqual(wf['sigmas']['inputs']['steps'],6); self.assertEqual(wf['lora']['inputs']['lora_name'],handler.LORAS[8])
        finally: handler._upload=old
    def test_rejects_non_24_fps(self):
        with self.assertRaises(ValueError): handler._params({'params':{'prompt':'x','fps':30}})

if __name__=='__main__': unittest.main()
