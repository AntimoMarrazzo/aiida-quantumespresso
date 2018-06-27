# -*- coding: utf-8 -*-
from aiida.orm import Code
from aiida.orm.data.base import Str, Float, Bool
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.bands import BandsData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.utils import WorkflowFactory
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')

class CustomPwBandStructureWorkChain(WorkChain):
    """
    Workchain to relax and compute the band structure for a given input structure
    using Quantum ESPRESSO's pw.x
    """
    def __init__(self, *args, **kwargs):
        super(CustomPwBandStructureWorkChain, self).__init__(*args, **kwargs)

    @classmethod
    def define(cls, spec):
        super(CustomPwBandStructureWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('pseudo_family', valid_type=Str)
        spec.input('protocol', valid_type=Str, default=Str('standard'))
        spec.outline(
            cls.setup_protocol,
            cls.setup_kpoints,
            cls.setup_parameters,
            cls.run_bands,
            cls.run_results,
        )
        spec.output('primitive_structure', valid_type=StructureData)
        spec.output('seekpath_parameters', valid_type=ParameterData)
        spec.output('scf_parameters', valid_type=ParameterData)
        spec.output('band_parameters', valid_type=ParameterData)
        spec.output('band_structure', valid_type=BandsData)

    def setup_protocol(self):
        """
        Setup of context variables and inputs for the PwBandsWorkChain. Based on the specified
        protocol, we define values for variables that affect the execution of the calculations
        """
        self.ctx.inputs = {
            'code': self.inputs.code,
            'structure': self.inputs.structure,
            'pseudo_family': self.inputs.pseudo_family,
            'parameters': {},
            'settings': {},
            'options': {
                'resources': {
                    'num_machines': 1
                },
                'max_wallclock_seconds': 1800,
            },
        }

        if self.inputs.protocol == 'standard':
            self.report('running the workchain in the "{}" protocol'.format(self.inputs.protocol.value))
            self.ctx.protocol = {
                'kpoints_mesh_offset': [0., 0., 0.],
                'kpoints_mesh_density': 0.2,
                'convergence_threshold': 1.E-10,
                'smearing': 'marzari-vanderbilt',
                'degauss': 0.02,
                'occupations': 'smearing',
                'tstress': True,
                'pseudo_data': {
			 'Ag': {'cutoff': 50.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Al': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Ar': {'cutoff': 60.0, 'dual': 4.0, 'pseudo': 'SG15-1.1'},
			 'As': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': '031US'},
			 'Au': {'cutoff': 45.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'B': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Ba': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Be': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Bi': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Br': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'C': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Ca': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Cd': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': '031US'},
			 'Ce': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Cl': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Co': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Cr': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.5'},
			 'Cs': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Cu': {'cutoff': 55.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Dy': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Er': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Eu': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'F': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Fe': {'cutoff': 90.0, 'dual': 12.0, 'pseudo': '031PAW'},
			 'Ga': {'cutoff': 70.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Gd': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Ge': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'H': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': '100US'},
			 'He': {'cutoff': 50.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Hf': {'cutoff': 50.0, 'dual': 4.0, 'pseudo': 'Dojo'},
			 'Hg': {'cutoff': 50.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Ho': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'I': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': '031PAW'},
			 'In': {'cutoff': 50.0, 'dual': 8.0, 'pseudo': '031US'},
			 'Ir': {'cutoff': 55.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'K': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Kr': {'cutoff': 45.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'La': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Li': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Lu': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Mg': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': '031PAW'},
			 'Mn': {'cutoff': 65.0, 'dual': 12.0, 'pseudo': 'GBRV-1.5'},
			 'Mo': {'cutoff': 35.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'N': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': 'THEOS'},
			 'Na': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.5'},
			 'Nb': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': '031PAW'},
			 'Nd': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Ne': {'cutoff': 50.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Ni': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'O': {'cutoff': 50.0, 'dual': 8.0, 'pseudo': '031PAW'},
			 'Os': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': '100US'},
			 'P': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': '100US'},
			 'Pb': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': '031PAW'},
			 'Pd': {'cutoff': 45.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Pm': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Po': {'cutoff': 75.0, 'dual': 8.0, 'pseudo': '100US'},
			 'Pr': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Pt': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Rb': {'cutoff': 30.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Re': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Rh': {'cutoff': 35.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Rn': {'cutoff': 120.0, 'dual': 8.0, 'pseudo': '100PAW'},
			 'Ru': {'cutoff': 35.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'S': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Sb': {'cutoff': 60.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Sc': {'cutoff': 40.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Se': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Si': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': '100US'},
			 'Sm': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Sn': {'cutoff': 60.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Sr': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Ta': {'cutoff': 45.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Tb': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Tc': {'cutoff': 30.0, 'dual': 4.0, 'pseudo': 'SG15'},
			 'Te': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Ti': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'Tl': {'cutoff': 50.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Tm': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'V': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.4'},
			 'W': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Xe': {'cutoff': 60.0, 'dual': 4.0, 'pseudo': 'SG15-1.1'},
			 'Y': {'cutoff': 35.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Yb': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'Wentzcovitch'},
			 'Zn': {'cutoff': 40.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
			 'Zr': {'cutoff': 30.0, 'dual': 8.0, 'pseudo': 'GBRV-1.2'},
                }
            }

    def setup_kpoints(self):
        """
        Define the k-point mesh for the relax and scf calculations. Also get the k-point path for
        the bands calculation for the initial input structure from SeeKpath
        """
        kpoints_mesh = KpointsData()
        kpoints_mesh.set_cell_from_structure(self.inputs.structure)
        kpoints_mesh.set_kpoints_mesh_from_density(
            distance=self.ctx.protocol['kpoints_mesh_density'],
            offset=self.ctx.protocol['kpoints_mesh_offset']
        )

        self.ctx.kpoints_mesh = kpoints_mesh

    def setup_parameters(self):
        """
        Setup the default input parameters required for the PwBandsWorkChain
        """
        structure = self.inputs.structure
        ecutwfc = []
        ecutrho = []

        for kind in structure.get_kind_names():
            try:
                dual = self.ctx.protocol['pseudo_data'][kind]['dual']
                cutoff = self.ctx.protocol['pseudo_data'][kind]['cutoff']
                cutrho = dual * cutoff
                ecutwfc.append(cutoff)
                ecutrho.append(cutrho)
            except KeyError as exception:
                self.abort_nowait('failed to retrieve the cutoff or dual factor for {}'.format(kind))

        natoms = len(structure.sites)
        conv_thr = self.ctx.protocol['convergence_threshold'] * natoms

        self.ctx.inputs['parameters'] = {
            'CONTROL': {
                'restart_mode': 'from_scratch',
                'tstress': self.ctx.protocol['tstress'],
            },
            'SYSTEM': {
                'ecutwfc': max(ecutwfc),
                'ecutrho': max(ecutrho),
                'smearing': self.ctx.protocol['smearing'],
                'degauss': self.ctx.protocol['degauss'],
                'occupations': self.ctx.protocol['occupations'],
            },
            'ELECTRONS': {
                'conv_thr': conv_thr,
            }
        }

    def run_bands(self):
        """
        Run the PwBandsWorkChain to compute the band structure
        """
        inputs = dict(self.ctx.inputs)

        options = inputs['options']
        settings = inputs['settings']
        parameters = inputs['parameters']
        #parameters['SYSTEM']['nbnd'] = 400
        # Final input preparation, wrapping dictionaries in ParameterData nodes
        inputs['kpoints_mesh'] = self.ctx.kpoints_mesh
        inputs['parameters'] = ParameterData(dict=parameters)
        inputs['settings'] = ParameterData(dict=settings)
        inputs['options'] = ParameterData(dict=options)
        inputs['relax'] = {
            'kpoints_distance': Float(self.ctx.protocol['kpoints_mesh_density']),
            'parameters': ParameterData(dict=parameters),
            'settings': ParameterData(dict=settings),
            'options': ParameterData(dict=options),
            'meta_convergence': Bool(False),
            'relaxation_scheme': Str('vc-relax'),
            'volume_convergence': Float(0.01)
        }
        inputs['workchain_options'] = ParameterData(dict = {'num_bands_factor':3.0}) 
        running = submit(PwBandsWorkChain, **inputs)

        self.report('launching PwBandsWorkChain<{}>'.format(running.pid))

        return ToContext(workchain_bands=running)

    def run_results(self):
        """
        Attach the relevant output nodes from the band calculation to the workchain outputs
        for convenience
        """
        self.report('workchain succesfully completed')

        for link_label in ['primitive_structure', 'seekpath_parameters', 'scf_parameters', 'band_parameters', 'band_structure']:
            if link_label in self.ctx.workchain_bands.out:
                node = self.ctx.workchain_bands.get_outputs_dict()[link_label]
                self.out(link_label, node)
                self.report("attaching {}<{}> as an output node with label '{}'"
                    .format(node.__class__.__name__, node.pk, link_label))
