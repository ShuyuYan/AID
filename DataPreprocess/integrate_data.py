import os
import pandas as pd

if __name__ == '__main__':
    nii_path = os.path.expanduser('~/Data/AID/428MRA')
    mra_path = os.path.expanduser('~/Data/AID/428mra.xlsx')
    baseline_path = os.path.expanduser('~/Data/AID/Baseline.xlsx')
    TA_list = os.listdir(nii_path)
    TA_list.sort()
    data = []
    head = ['TA', 'nii_path', 'site', 'name', 'pinyin', 'mra_examination_date', 'mra_examination_re_des_1',
            'dob', 'age_enrolled', 'sex',
            'first_manifestation', 'origin_symptom_date', 'diagnosis_date', 'renal_hypertension',
            'kidney_insufficiency',
            'heart_dysfunction', 'cerebral_infarction', 'kidney_atrophy', 'vision_loss', 'other_complication_1',
            'fever',
            'chest_pain', 'fatigue', 'amaurosis_symptoms', 'neck_pain', 'night_sweat', 'abdominal_pain', 'up_limb_cold',
            'low_limb_cold', 'up_limb_numb', 'low_limb_numb', 'head_pain_dizness', 'inter_claudication_signs',
            'joint_swell_pain', 'erythema_nodosum', 'other_rash', 'oral_ulcer', 'weight_decrease', 'uveitis',
            'visual_loss',
            'left_visual_loss', 'right_visual_loss', 'audition_loss', 'left_audition_loss', 'right_audition_loss',
            'other_symptom', 'other_sym_detail_descrip', 'Unnamed: 46', 'rig_carotid_artery', 'lef_carotid_artery',
            'rig_brachial_artery', 'lef_brachial_artery', 'rig_radial_artery', 'lef_radial_artery',
            'rig_femoral_artery',
            'lef_femoral_artery', 'rig_popliteal_artery', 'lef_popliteal_artery', 'rig_dorsal_plantar_artery',
            'lef_dorsal_plantar_artery', 'right_neck', 'left_neck', 'right_clavian', 'left_clavian', 'right_scapula',
            'left_scapula', 'right_kidney', 'left_kidney', 'epigastrium', 'cardiac_valve_area', 'lef_up_limb_sbp',
            'lef_up_limb_dbp', 'rig_up_limb_sbp', 'rig_up_limb_dbp', 'lef_low_limb_sbp', 'lef_low_limb_dbp',
            'rig_low_limb_sbp', 'rig_low_limb_dbp', 'Unnamed: 77', 'hb', 'wbc', 'plt', 'neutrophils_x10_9_l',
            'lymphocytes_x10_9_l', 'monocytes_x10_9_l', 'eosnophils_x10_9_l', 'basophils_x10_9_l',
            'the_average_volume_of_platpin', 'thrombocytocrit', 'platelet_large_cell_ratio',
            'platelet_distribution_width',
            'urine_rbc', 'urine_wbc', 'urine_protein', 'bun', 'cr', 'ua', 'glucose', 'tc', 'tg', 'hdl', 'ldl', 'apo_a',
            'apob',
            'apoe', 'lp_a', 'tbil', 'g', 'a', 'alt', 'ast', 'akp', 'gamma_gt', 'ldh', 'esr', 'crp', 'saa', 'ferritin',
            'igg',
            'igm', 'iga', 'ige', 'il1beta', 'il6', 'il8', 'il2r', 'il10', 'tnf', 'c3', 'c4', 'ch50',
            'prothrombin_time_pt',
            'prothrombin_time_ratio', 'international_normalized_r', 'd_dimer', 'fibrinogen', 'hbsag', 'hbsab', 'hbeag',
            'hbeab', 'hbcab（监测', 'hcv', 'ca', 'p', 'ana', 'dsdna', 'anua', 'acl', 'antib2gp1', 'anti_rnp', 'anti_sm',
            'anti_ssa', 'anti_ssb', 'anti_scl70', 'anti_jo1', 'anti_pm_scl', 'aca', 'anti_pcna', 'aha', 'rrnp', 'ama',
            'canca',
            'panca', 'pr3', 'mpo', 't_spot（1=阳性，0=阴性）', 'ctnt', 'nt_probnp', 'cd19', 'cd3', 'cd4', 'cd8', 'cd4_cd8',
            'nkc',
            'lymphocytes', 'b_cell_no', 't_cell_no', 'th_cell_no', 'ts_cell_no', 'nk_cell_no', 'cmv_igg', 'cmv_igm',
            'ebv_iga',
            'ebv_igm', 'Kerr评分', '影像学分型', 'type（1-GC+cs，2-GC+b/ts，3-others（单GC or GC+中药/HCQ，4-上述药物均无）',
            'Effect 1=CR有效，0=无效']

    mra = pd.read_excel(mra_path, sheet_name='Report')
    bl = pd.read_excel(baseline_path, sheet_name='Baseline', skiprows=2)

    for i in range(len(TA_list)):
        data.append([TA_list[i], nii_path + '/' + TA_list[i]])
        for j in range(len(mra)):
            if mra['serial_number'][j] == TA_list[i]:
                result = []
                for col in head[2:7]:
                    cell_value = mra.iloc[j][col]
                    result.append(str(cell_value))
                data[i] += result
                break

        for j in range(len(bl)):
            if bl['serial_number'][j] == TA_list[i]:
                result = []
                for col in head[7:]:
                    cell_value = bl.iloc[j][col]
                    result.append(cell_value)
                data[i] += result
                break

    save = pd.DataFrame(data=data, columns=head)
    save.to_excel(os.path.expanduser('~/Data/AID/all.xlsx'), sheet_name='data')
