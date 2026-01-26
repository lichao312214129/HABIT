"""
DICOM 匿名化脚本
用于移除 DICOM 文件中的患者标识信息，同时保留文件夹层次结构
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import pydicom
from pydicom.dataset import Dataset


def anonymize_dicom_file(input_file: Path, output_file: Path, keep_tags: list = None) -> bool:
    """
    匿名化单个 DICOM 文件
    
    Args:
        input_file: 输入 DICOM 文件路径
        output_file: 输出 DICOM 文件路径
        keep_tags: 需要保留的标签列表（可选）
    
    Returns:
        bool: 是否成功
    """
    try:
        ds = pydicom.dcmread(input_file)
        
        # 需要移除的标签（患者标识信息）
        tags_to_remove = [
            'PatientName',
            'PatientID',
            'PatientBirthDate',
            'PatientAddress',
            'PatientMotherBirthName',
            'PatientSex',
            'OtherPatientIDs',
            'OtherPatientNames',
            'PatientBirthName',
            'PatientSize',
            'PatientWeight',
            'PatientAge',
            'PatientComments',
            'PatientState',
            'PatientTelecoms',
            'PatientInsurancePlanCodeSequence',
            'PatientPrimaryLanguageCodeSequence',
            'PatientPrimaryLanguageModifierCodeSequence',
            'EthnicGroup',
            'Occupation',
            'AdditionalPatientHistory',
            'ResponsiblePerson',
            'ResponsibleOrganization',
            'ReferringPhysicianName',
            'ReadingPhysicianName',
            'RequestingPhysician',
            'NameOfPhysiciansReadingStudy',
            'OperatorsName',
            'PerformingPhysicianName',
            'InstitutionName',
            'InstitutionAddress',
            'InstitutionalDepartmentName',
            'StationName',
            'DeviceSerialNumber',
            'SoftwareVersions',
            'StudyDescription',
            'SeriesDescription',
            'ImageComments',
            'AcquisitionComments',
            'FrameComments',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
        ]
        
        # 移除标签
        for tag in tags_to_remove:
            if tag in ds:
                del ds[tag]
        
        # 添加匿名化标记
        ds.Anonymized = 'Yes'
        ds.AnonymizationDate = datetime.now().strftime('%Y%m%d')
        
        # 保存匿名化后的文件
        ds.save_as(output_file, write_like_original=False)
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def anonymize_dicom_directory(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = True,
    overwrite: bool = False
) -> dict:
    """
    匿名化目录中的所有 DICOM 文件，保留文件夹层次结构
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        recursive: 是否递归处理子目录
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        dict: 处理统计信息
    """
    stats = {
        'total_files': 0,
        'success_files': 0,
        'failed_files': 0,
        'skipped_files': 0
    }
    
    # 遍历所有 DICOM 文件
    pattern = '**/*.dcm' if recursive else '*.dcm'
    
    for dcm_file in input_dir.glob(pattern):
        if not dcm_file.is_file():
            continue
        
        stats['total_files'] += 1
        
        # 计算相对路径
        rel_path = dcm_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # 检查文件是否已存在
        if output_file.exists():
            if overwrite:
                print(f"Overwriting: {output_file}")
            else:
                print(f"Skipping (exists): {output_file}")
                stats['skipped_files'] += 1
                continue
        
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 匿名化文件
        if anonymize_dicom_file(dcm_file, output_file):
            stats['success_files'] += 1
            print(f"✓ Anonymized: {rel_path}")
        else:
            stats['failed_files'] += 1
            print(f"✗ Failed: {rel_path}")
    
    return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DICOM 匿名化工具 - 移除患者标识信息，保留文件夹结构'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入 DICOM 目录路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='递归处理子目录（默认：True）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的文件'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # 验证输入目录
    if not input_dir.exists():
        print(f"Error: 输入目录不存在: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: 输入路径不是目录: {input_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DICOM 匿名化工具")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"递归处理: {'是' if args.recursive else '否'}")
    print(f"覆盖文件: {'是' if args.overwrite else '否'}")
    print(f"{'='*60}\n")
    
    # 执行匿名化
    stats = anonymize_dicom_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite
    )
    
    # 显示统计信息
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"{'='*60}")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功: {stats['success_files']}")
    print(f"失败: {stats['failed_files']}")
    print(f"跳过: {stats['skipped_files']}")
    print(f"{'='*60}\n")
    
    if stats['failed_files'] > 0:
        print(f"警告: {stats['failed_files']} 个文件处理失败，请检查错误信息")


if __name__ == '__main__':
    main()
