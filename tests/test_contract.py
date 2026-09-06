import base64, json, pathlib, unittest
from unittest import mock
import handler

class ContractTests(unittest.TestCase):
    def test_defaults_and_frames(self):
        p=handler._params({'params':{'prompt':'test'}})
        self.assertEqual(p['steps'],8); self.assertEqual(p['fps'],24); self.assertEqual(handler._frames(15),362)
    def test_workflow_injects_fp8_params(self):
        inp={'params':{'prompt':'a test','steps':6,'seed':7},'images':[{'name':'x.png','data':base64.b64encode(b'not-an-image').decode()}]}
        old=handler._upload; handler._upload=lambda item:'x.png'
        try:
            wf=handler._workflow(inp,handler._params(inp)); self.assertEqual(wf['unet']['inputs']['unet_name'],'minimax_h3_fl2va_mxfp8.safetensors'); self.assertEqual(wf['sigmas']['inputs']['steps'],6); self.assertEqual(wf['lora']['inputs']['lora_name'],handler.LORAS[8]); self.assertIn('_768p_', wf['lora']['inputs']['lora_name'])
        finally: handler._upload=old

    def test_legacy_subgraph_models_are_normalized_once(self):
        workflow = {
            '105': {'class_type': 'MiniMaxH3', 'inputs': {
                'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
                'clip_name': 'undefined',
                'vae_name': 'undefined',
                'audio_vae': 'undefined',
                'lora_name': 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
            }},
            '105:6': {'class_type': 'UNETLoader', 'inputs': {
                'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
            }},
            '105:121': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                'lora_name': 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
            }},
        }
        decoded = handler._decode_workflow(json.dumps(workflow))
        handler._normalize_workflow_models(decoded)
        self.assertEqual(decoded['105']['inputs']['unet_name'], 'minimax_h3_ref2va_pruned_fp8_scaled.safetensors')
        self.assertEqual(decoded['105']['inputs']['clip_name'], handler.CLIP_MODEL)
        self.assertEqual(decoded['105']['inputs']['vae_name'], handler.VIDEO_VAE)
        self.assertEqual(decoded['105']['inputs']['audio_vae'], handler.AUDIO_VAE)
        self.assertEqual(decoded['105']['inputs']['lora_name'], handler.REF_LORA)
        self.assertEqual(decoded['105:6']['inputs']['unet_name'], 'minimax_h3_ref2va_pruned_fp8_scaled.safetensors')
        self.assertEqual(decoded['105:121']['inputs']['lora_name'], handler.REF_LORA)

    def test_string_workflow_is_decoded(self):
        workflow = {'unet': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
        }}}
        self.assertEqual(handler._decode_workflow(json.dumps(workflow)), workflow)
        with self.assertRaisesRegex(ValueError, 'malformed JSON'):
            handler._decode_workflow('{"a":1} trailing')
    def test_rejects_non_24_fps(self):
        with self.assertRaises(ValueError): handler._params({'params':{'prompt':'x','fps':30}})

    def test_unified_asset_groups_keep_legacy_images(self):
        groups = handler._asset_groups({'images': [{'name': 'a.png'}], 'assets': [
            {'type': 'video', 'name': 'motion.mp4'}, {'type': 'audio', 'name': 'voice.wav'}]})
        self.assertEqual([x['name'] for x in groups['image']], ['a.png'])
        self.assertEqual([x['name'] for x in groups['video']], ['motion.mp4'])
        self.assertEqual([x['name'] for x in groups['audio']], ['voice.wav'])

    def test_supported_modes(self):
        for mode in ('t2v', 'i2v', 'fl2v', 'r2v', 'v2v', 'rv2v'):
            p = handler._params({'params': {'prompt': 'test', 'mode': mode}})
            self.assertEqual(p['mode'], mode)

    def test_user_facing_mode_aliases(self):
        aliases = {
            'image_to_video': 'i2v',
            'video_to_video': 'v2v',
            'image_video_mix': 'rv2v',
            'audio_reference': 'r2v',
        }
        for source, expected in aliases.items():
            p = handler._params({'params': {'prompt': 'test', 'mode': source}})
            self.assertEqual(p['mode'], expected)

    def test_canvas_workflow_requires_api_export(self):
        inp = {'workflow': {'nodes': [], 'links': []},
               'params': {'prompt': 'test', 'mode': 'i2v'}}
        with self.assertRaisesRegex(ValueError, r'Save \(API Format\)'):
            handler._workflow(inp, handler._params(inp))

    def test_custom_feature_workflow_is_passthrough(self):
        inp = {
            'workflow': {
                'easy': {'class_type': 'MiniMaxH3Easy', 'inputs': {'prompt': 'desktop'}},
                'image': {'class_type': 'LoadImage', 'inputs': {'image': 'input.png'}},
            },
            'params': {'prompt': 'desktop', 'mode': 'r2v'},
            'assets': [{'type': 'image', 'name': 'character.png', 'data': base64.b64encode(b'fake').decode()}],
        }
        old = handler._upload
        handler._upload = lambda item: item['name']
        try:
            wf = handler._workflow(inp, handler._params(inp))
        finally:
            handler._upload = old
        self.assertEqual(wf['easy']['class_type'], 'MiniMaxH3Easy')

    def test_incomplete_requests_are_rejected_before_rendering(self):
        result = handler.handler({'id': 'job-1', 'input': {}})
        self.assertEqual(result['type'], 'ValueError')
        self.assertIn('incomplete request', result['error'])

    def test_version_probe_stays_on_compatibility_path(self):
        old = handler._run_runonrunpod
        handler._run_runonrunpod = lambda job: {'status': 'ok', 'worker_version': 'h3'}
        try:
            result = handler.handler({'id': 'job-2', 'input': {'action': 'version'}})
        finally:
            handler._run_runonrunpod = old
        self.assertEqual(result['status'], 'ok')

    def test_live_model_schema_rejects_wrong_profile_model(self):
        schema = {
            'UNETLoader': {'input': {'required': {'unet_name': [['minimax_h3_fl2va_mxfp8.safetensors']]}}},
            'CLIPLoader': {'input': {'required': {'clip_name': [['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']]}}},
        }
        workflow = {
            'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'}},
            'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}},
        }
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = json.dumps(schema).encode()
        with mock.patch('urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                handler._validate_workflow_before_prompt(workflow)

    def test_live_model_schema_accepts_other_profile_model(self):
        schema = {
            'UNETLoader': {'input': {'required': {'unet_name': [['minimax_h3_fl2va_pruned_int8_convrot.safetensors']]}}},
            'CLIPLoader': {'input': {'required': {'clip_name': [['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']]}}},
        }
        workflow = {
            'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'}},
            'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}},
        }
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = json.dumps(schema).encode()
        with mock.patch('urllib.request.urlopen', return_value=response):
            handler._validate_workflow_before_prompt(workflow)

    def test_media_loader_missing_input_is_rejected(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{}'
        workflow = {'image': {'class_type': 'LoadImage', 'inputs': {}}}
        with mock.patch('urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'missing required input image'):
                handler._validate_workflow_before_prompt(workflow)

if __name__=='__main__': unittest.main()
